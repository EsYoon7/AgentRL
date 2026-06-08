import asyncio

import os
import torch
import torch.distributed as dist

from ..utils import init_custom_process_group
from ..utils.torch_patch import broadcast_object_list
from ..workers.abstract import AbstractTrainWorker, AbstractAsyncRolloutWorker

TensorBuffer = list[tuple[str, torch.Tensor]]


def send_buffer(buffer: TensorBuffer, pg):
    descriptions = {k: (v.shape, v.dtype) for k, v in buffer}
    lst = [descriptions]
    broadcast_object_list(lst, group_src=0, group=pg)
    for _, v in buffer:
        dist.broadcast(v, group_src=0, group=pg)


def receive_buffer(pg, device=None) -> TensorBuffer:
    buffer = []
    lst: list[None | dict] = [None]
    broadcast_object_list(lst, group_src=0, group=pg, device=device)
    descriptions: dict = lst[0]
    for k, (shape, dtype) in descriptions.items():
        v = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(v, group_src=0, group=pg)
        buffer.append((k, v))
    return buffer


# asyncio.to_thread로 넘기는 NCCL 호출은 매번 다른 워커 스레드에서 실행된다.
# CUDA current device는 스레드-로컬이므로, 호출 직전 그 스레드에서 device를
# receiver의 self.device로 맞춰준다 (하드코딩이 아니라 __init__에서 계산된 값).
def _barrier_on(pg, device):
    torch.cuda.set_device(device)
    dist.barrier(group=pg)


def _receive_buffer_on(pg, device):
    torch.cuda.set_device(device)
    return receive_buffer(pg, device)


def _broadcast_object_list_on(lst, group_src, pg, device):
    torch.cuda.set_device(device)
    broadcast_object_list(lst, group_src=group_src, group=pg, device=device)


class NCCLTensorSender:
    # def __init__(self, worker: AbstractTrainWorker, addr, port, world_size):
    #     self.worker = worker
    #     self.worker_rank = worker.rank
    #     print(f"sender {torch.cuda.device_count()=} {addr=} {port=} {world_size=} {worker.rank=}")
    #     if self.worker_rank == 0:
    #         sender_device = torch.device("cuda:0")
    #         # PyTorch 2.6+ does an eager NCCL handshake against `device_id`; if the
    #         # current CUDA context has not been pinned to that device yet the call
    #         # raises `Cuda failure 'invalid argument'`. Set device explicitly here.
    #         torch.cuda.set_device(sender_device)
    #         # NOTE: do not pass `device_id` — PyTorch 2.6+ would otherwise call
    #         # `eager_connect_single_device`, which trips a UnicodeDecodeError on
    #         # the binary NCCL unique-ID exchanged through this custom TCP store.
    #         # Lazy connect on first collective is fine here.
    #         self.pg = init_custom_process_group(
    #             backend="nccl",
    #             init_method=f"tcp://{addr}:{port}",
    #             world_size=world_size,
    #             rank=0,
    #             group_name=f"nccl_comm_{addr}_{port}",
    #             device_id=None,
    #         )
    def __init__(self, worker, addr, port, world_size):
        self.worker = worker
        self.worker_rank = worker.rank
        print(f"[SENDER INIT] rank={worker.rank} addr={addr} port={port} ws={world_size}", flush=True)
        if self.worker_rank == 0:
            print(f"[SENDER INIT r0] creating PG server on {addr}:{port} ...", flush=True)
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            torch.cuda.set_device(self.device)
            self.pg = init_custom_process_group(
                backend="nccl",
                init_method=f"tcp://{addr}:{port}",
                world_size=world_size,
                rank=0,
                group_name=f"nccl_comm_{addr}_{port}",
                device_id=None,
            )
            print(f"[SENDER INIT r0] PG server created", flush=True)

    def send(self, bucket_size):
        print(f"[SEND r{self.worker_rank}] enter", flush=True)
        n = 0

        # 1단계: 전 rank가 param_generator(PG2 = FSDP all-gather)를 끝까지 함께 돈다.
        #         streamer(self.pg) 통신은 절대 이 루프 안에서 하지 않는다.
        #         rank0만 결과를 보관하되 GPU OOM을 피하려고 CPU로 옮겨 둔다.
        param_device = None
        collected: TensorBuffer = []
        for key, val in self.worker.param_generator():
            n += 1
            if self.worker_rank == 0:
                if param_device is None:
                    param_device = val.device   
                collected.append((key, val.detach().to("cpu", copy=True)))
            del val
        print(f"[SEND r{self.worker_rank}] param_generator done, n={n}", flush=True)

        # 2단계: rank0만 streamer로 전송. r1/r2는 여기서 할 일이 없다.
        if self.worker_rank == 0:

            d = torch.cuda.current_device()
            try:
                u = torch.cuda.get_device_properties(d).uuid
            except Exception:
                u = "n/a"
            libs = []
            for line in open(f"/proc/{os.getpid()}/maps"):
                if "nccl" in line.lower():
                    p = line.split()[-1]
                    if p not in libs:
                        libs.append(p)
            print(f"[NCCLCHK pid={os.getpid()}] CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"torch_nccl={torch.cuda.nccl.version()} cur={d} uuid={u} libnccl={libs}", flush=True)
            
            print(f"[SEND r0] before streamer barrier, "
                  f"dev={torch.cuda.current_device()} collected={len(collected)}", flush=True)
            print(f"[SEND r0] param_device={param_device} "
                    f"init_set_device=cuda:0 cur={torch.cuda.current_device()}", flush=True)
            torch.cuda.set_device(self.device)
            dist.barrier(group=self.pg)

            buffer: TensorBuffer = []
            size = 0
            for key, val in collected:
                v = val.to(self.device)            # 전송 직전 GPU로 복원
                buffer.append((key, v))
                size += v.numel() * v.element_size()
                if size >= bucket_size:
                    send_buffer(buffer, self.pg)
                    broadcast_object_list([False], group_src=0, group=self.pg)
                    buffer = []
                    size = 0
            # 남은 버킷 flush + 종료 신호
            send_buffer(buffer, self.pg)
            broadcast_object_list([True], group_src=0, group=self.pg)

        print(f"[SEND r{self.worker_rank}] done, total_params={n}", flush=True)


class NCCLTensorReceiver:
    def __init__(self, worker: AbstractAsyncRolloutWorker, addr, port, world_size, offset):
        self.worker = worker
        # TODO: temporary hack for multiple devices in one process.
        self.device = torch.device(
            f"cuda:{(offset + worker.rank) % torch.cuda.device_count()}"
        )
        print(f"[RECV INIT] worker.rank={worker.rank} offset={offset} "
              f"device_count={torch.cuda.device_count()} -> device={self.device} "
              f"world_size={world_size} addr={addr} port={port}", flush=True)
        # See NCCLTensorSender for the rationale: pin the CUDA context to the
        # target device before PyTorch's eager NCCL handshake runs, and pass
        # `device_id=None` to skip eager_connect_single_device entirely (it
        # raises UnicodeDecodeError on the binary unique-ID round-trip).
        torch.cuda.set_device(self.device)
        self.pg = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{addr}:{port}",
            world_size=world_size,
            rank=offset + worker.rank,
            group_name=f"nccl_comm_{addr}_{port}",
            device_id=None,
        )
        d = torch.cuda.current_device()
        try:
            u = torch.cuda.get_device_properties(d).uuid
        except Exception:
            u = "n/a"
        libs = []
        for line in open(f"/proc/{os.getpid()}/maps"):
            if "nccl" in line.lower():
                p = line.split()[-1]
                if p not in libs:
                    libs.append(p)
        print(f"[NCCLCHK pid={os.getpid()}] CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"torch_nccl={torch.cuda.nccl.version()} cur={d} uuid={u} libnccl={libs}", flush=True)

    async def async_receive(self):
        print(f"[RECV] before barrier, rank={self.worker.rank}, device={self.device}", flush=True)
        # sender와 정확히 한 번 rendezvous. barrier 개수는 sender와 일치해야 한다.
        await asyncio.to_thread(_barrier_on, self.pg, self.device)
        print(f"[RECV] after barrier", flush=True)

        done = False
        await self.worker.async_acquire_writer_lock()
        task = asyncio.sleep(0)
        while not done:
            buffer = await asyncio.to_thread(_receive_buffer_on, self.pg, self.device)
            await task
            task = asyncio.create_task(self.worker.update_params(buffer))

            lst: list[bool | None] = [None]
            await asyncio.to_thread(_broadcast_object_list_on, lst, 0, self.pg, self.device)
            done = lst[0]
        await task
        await self.worker.async_release_writer_lock()

class NCCLTensorSenderDist:
    """Trainer 측 가중치 broadcaster (distributed weight sync).

    모든 train rank가 param_generator()를 함께 돈다(메인 PG에서 FSDP all_gather).
    rank0만 weight-update group에도 속해, 언샤딩된 각 파라미터를 param_generator
    순서대로 SGLang에 broadcast한다. SGLang은 동일 순서의 names/dtypes/shapes로
    update_weights_from_distributed를 통해 받는다.
    """

    def __init__(self, worker, addr, port, world_size, group_name="weight_update_group"):
        self.worker = worker
        self.world_size = world_size
        self.group_name = group_name
        self.device = None
        self.pg = None
        if worker.rank == 0:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            torch.cuda.set_device(self.device)
            print(f"[SENDER r0] creating weight-update PG {addr}:{port} ws={world_size}", flush=True)
            self.pg = init_custom_process_group(
                backend="nccl",
                init_method=f"tcp://{addr}:{port}",
                world_size=world_size,
                rank=0,
                group_name=group_name,
                device_id=None,   # eager NCCL handshake 회피 (lazy connect)
            )
            print(f"[SENDER r0] weight-update PG ready", flush=True)

    def collect_meta(self):
        """전 rank가 param_generator를 1회 돌고(all_gather 참여), rank0이
        param_generator 순서의 (names, dtypes, shapes)를 반환. 텐서는 버린다.
        모델 구조라 불변이므로 셋업에서 한 번만 호출한다."""
        names, dtypes, shapes = [], [], []
        for k, v in self.worker.param_generator():
            if self.worker.rank == 0:
                names.append(k)
                dtypes.append(str(v.dtype).split(".")[-1])   # "bfloat16"
                shapes.append(list(v.shape))
            del v
        if self.worker.rank == 0:
            print(f"[SENDER] collected metadata for {len(names)} params", flush=True)
            return names, dtypes, shapes
        return None

    def send(self):
        """param_generator 순서대로 모든 파라미터를 broadcast.
        rank0만 broadcast하고, rank1..N은 param_generator의 all_gather만 참여한다.
        (한 번에 하나씩 broadcast → train 측 메모리는 파라미터 1개분만 점유)"""
        n = 0
        if self.worker.rank == 0:
            torch.cuda.set_device(self.device)
        for k, v in self.worker.param_generator():
            n += 1
            if self.worker.rank == 0:
                dist.broadcast(v.to(self.device), src=0, group=self.pg)
            del v
        if self.worker.rank == 0:
            torch.cuda.empty_cache()
        print(f"[SEND r{self.worker.rank}] done, params={n}", flush=True)