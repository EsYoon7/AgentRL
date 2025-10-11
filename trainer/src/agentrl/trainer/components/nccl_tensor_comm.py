import asyncio

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


def receive_buffer(pg) -> TensorBuffer:
    buffer = []
    lst: list[None | dict] = [None]
    broadcast_object_list(lst, group_src=0, group=pg)
    descriptions: dict = lst[0]
    for k, (shape, dtype) in descriptions.items():
        v = torch.empty(shape, dtype=dtype, device='cuda')
        dist.broadcast(v, group_src=0, group=pg)
        buffer.append((k, v))
    return buffer


class NCCLTensorSender:
    def __init__(self, worker: AbstractTrainWorker, addr, port, world_size):
        self.worker = worker
        self.worker_rank = worker.rank
        print(f"sender {torch.cuda.device_count()=} {addr=} {port=} {world_size=} {worker.rank=}")
        if self.worker_rank == 0:
            self.pg = init_custom_process_group(
                backend="nccl",
                init_method=f"tcp://{addr}:{port}",
                world_size=world_size,
                rank=0,
                group_name=f"nccl_comm_{addr}_{port}",
            )

    def send(self, bucket_size):
        buffer: TensorBuffer = []
        size = 0
        if self.worker_rank == 0:
            dist.barrier(group=self.pg)
        for item in self.worker.param_generator():
            key, val = item
            size += val.numel() * val.element_size()
            if size >= bucket_size:
                if self.worker_rank == 0:
                    # broadcast tensor
                    send_buffer(buffer, self.pg)

                    # broadcast done
                    lst = [False]
                    broadcast_object_list(lst, group_src=0, group=self.pg)

                # clear buffer
                del buffer
                buffer = []
                size = 0
            else:
                buffer.append((key, val))

        if self.worker_rank == 0:
            send_buffer(buffer, self.pg)
            # broadcast done
            lst = [True]
            broadcast_object_list(lst, group_src=0, group=self.pg)


class NCCLTensorReceiver:
    def __init__(self, worker: AbstractAsyncRolloutWorker, addr, port, world_size, offset):
        self.worker = worker
        self.pg = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{addr}:{port}",
            world_size=world_size,
            rank=offset + worker.rank,
            group_name=f"nccl_comm_{addr}_{port}",
        )

    async def async_receive(self):
        await asyncio.to_thread(dist.barrier, group=self.pg)
        done = False
        await self.worker.async_acquire_writer_lock()
        task = asyncio.sleep(0)
        while not done:
            buffer = await asyncio.to_thread(receive_buffer, self.pg)
            await task
            task = asyncio.create_task(self.worker.update_params(buffer))

            lst: list[bool | None] = [None]
            await asyncio.to_thread(broadcast_object_list, lst, group_src=0, group=self.pg)
            done: bool = lst[0]
        await task
        await self.worker.async_release_writer_lock()
