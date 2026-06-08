import argparse
import asyncio
import os
import threading
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import cycle
from pathlib import Path

import ray
import torch
import wandb
import yaml
import math
from ray.util import placement_group
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoProcessor

from agentrl.trainer.agentic.data_provider import get_agentic_datasets
from agentrl.trainer.algorithms.advantage import compute_advantage
from agentrl.trainer.algorithms.loss_funcs import log_prob_loss, ppo_loss
from agentrl.trainer.algorithms.metrics import calc_metrics, calc_batch_rl_metrics, calc_data_metrics, calc_adv_metrics
from agentrl.trainer.components.nccl_tensor_comm import NCCLTensorSenderDist
from agentrl.trainer.components.timer import Timer
from agentrl.trainer.utils import append_with_prefix, reduce_dict, pretty_print_metrics, repeat, interleave
from agentrl.trainer.workers.collective_handle import spawn
from agentrl.trainer.workers.fsdp_worker import FSDPWorker
from agentrl.trainer.workers.fsdp_worker_mm import FSDPWorkerMM

# === multimodal additions ===
from agentrl.trainer.workers.async_sglang_worker_mm import AsyncSglangWorkerMM
from agentrl.trainer.components.multimodal_task import multimodal_chat_task
from agentrl.trainer.components.task_manager_mm import DistributedTaskManagerMM as DistributedTaskManager


WEIGHT_GROUP = "weight_update_group"


def _as_list(x):
    return x if isinstance(x, list) else [x]


def collect_val_metrics(val_task_manager, event_loop):
    results = asyncio.run_coroutine_threadsafe(
        val_task_manager.get_all(), event_loop
    ).result()
    return gather_metrics(results)


def gather_metrics(data):
    by_source = defaultdict(list)
    for item in data:
        by_source[item["data_source"]].append(item)

    metrics = {}
    over_all_metrics = defaultdict(list)
    for source, items in by_source.items():
        source_metrics = {**calc_metrics(items), **calc_batch_rl_metrics(items)}
        if "advantages" in items[0]:
            source_metrics.update(calc_adv_metrics(items))
        append_with_prefix(metrics, f"{source}/", source_metrics)
        for k, v in source_metrics.items():
            over_all_metrics[k].append(v)
    overall_metrics = {k: sum(v) / len(v) for k, v in over_all_metrics.items()}
    append_with_prefix(metrics, "overall/", overall_metrics)
    return metrics


def main(config):
    # spawn workers
    rollout_config = config.get("rollout", {})
    actor_config = config.get("actor", {})
    ref_config = deepcopy(actor_config)
    ref_config.update(config.get("ref", {}))
    task_config = config.get("task", {})

    total_gpus = int(ray.cluster_resources()["GPU"])
    rollout_gpus = int(config.get("rollout_ratio", 0.5) * total_gpus)
    actor_gpus = total_gpus - rollout_gpus
    print(f"Total GPUs: {total_gpus}, Rollout GPUs: {rollout_gpus}, Actor GPUs: {actor_gpus}")

    rollout_tp = rollout_config.get("tp_size", 1)
    rollout_placement = placement_group([{"CPU": 1, "GPU": rollout_tp}] * (rollout_gpus // rollout_tp))
    actor_ref_placement = placement_group([{"CPU": 1, "GPU": 1}] * actor_gpus)

    rollout = spawn(AsyncSglangWorkerMM, rollout_placement, num_gpus=rollout_tp)(rollout_config)
    actor = spawn(FSDPWorkerMM, actor_ref_placement, num_gpus=0.5)(actor_config)
    ref = spawn(FSDPWorker, actor_ref_placement, num_gpus=0.5)(ref_config)

    # weight-update group의 rendezvous 주소/포트 (메인 train PG와 다른 free 포트)
    streamer_ip, streamer_port = ray.get(actor.dispatch_rank0().get_addr_and_port())
    # weight group 멤버 = trainer rank0(src) + SGLang tp 랭크들
    weight_world_size = 1 + rollout_tp

    rollout.build_engine(config["model_path"])

    actor.build_model(config["model_path"])
    actor.build_optimizer()
    actor.build_checkpoint_manager()

    # === distributed weight-sync 셋업 (NCCL broadcast; IPC 대체) ===
    # weight group 양끝(SGLang receiver rank_offset=1.., trainer rank0 src)을
    # 동시에 issue해서 NCCL rendezvous시킨다. 순차 ray.get 하면 deadlock.
    _sg_init = rollout.init_weight_update_group(
        streamer_ip, streamer_port,
        rank_offset=1, world_size=weight_world_size, group_name=WEIGHT_GROUP,
    )
    _snd_reg = actor.register_plugin(
        "param_sender", NCCLTensorSenderDist,
        streamer_ip, streamer_port, weight_world_size, WEIGHT_GROUP,
    )
    ray.get(_as_list(_sg_init))
    ray.get(_as_list(_snd_reg))

    # param_generator 순서의 정적 메타데이터(names/dtypes/shapes) 1회 수집.
    # SGLang이 매 sync마다 올바른 텐서를 받기 위해 사용.
    _metas = ray.get(actor.call_plugin("param_sender", "collect_meta"))
    param_names, param_dtypes, param_shapes = next(m for m in _metas if m is not None)
    print(f"[trainer] weight-sync ready: {len(param_names)} params, "
          f"group ws={weight_world_size}", flush=True)

    def sync_weights():
        """distributed weight sync: trainer rank0가 전 파라미터를 broadcast,
        SGLang이 update_weights_from_distributed로 수신. 둘을 동시 issue 후 함께 대기."""
        recv = rollout.update_params(param_names, param_dtypes, param_shapes)
        send = actor.call_plugin("param_sender", "send")
        ray.get(_as_list(recv) + _as_list(send))

    # micro-batch 분배 단위(=actor FSDP mesh size).
    train_world = ray.get(actor.dispatch_rank0().get_data_parallel_size())
    print(f"[trainer] actor data-parallel size = {train_world}", flush=True)

    ref.build_model(config["model_path"])

    # prepare datasets
    train_names = task_config.get("train_tasks", [])
    val_names = task_config.get("val_tasks", [])
    train_datasets = get_agentic_datasets(train_names, task_config["base_url"])
    val_datasets = get_agentic_datasets(val_names, task_config["base_url"])

    # prepare task workers
    train_config = config["train"]
    val_config = config["val"]
    n = train_config["n"]
    concurrency = train_config["concurrency"]
    batch_size = train_config["batch_size"]
    real_bsz = batch_size * n
    val_concurrency = val_config["concurrency"]

    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    def run_loop():
        asyncio.set_event_loop(event_loop)
        event_loop.run_forever()

    async_thread = threading.Thread(target=run_loop, daemon=True)
    async_thread.start()

    tokenizer = AutoProcessor.from_pretrained(config["model_path"])

    train_task_manager = DistributedTaskManager(
        task_fn=lambda item: multimodal_chat_task(
            item,
            config=task_config,
            tokenizer=tokenizer,
            gen_fn=partial(
                rollout.dispatch_rank(hash(str(item)) % rollout.world_size).generate_with_ids,
                sampling_params=train_config.get("sampling_params", {}),
            ),
        ),
        max_queue_size=real_bsz * 2,
        max_buffer_size=real_bsz * 2,
        buffer_group_size=1,
        num_workers=concurrency,
        event_loop=event_loop,
    )
    train_task_manager.start()

    # resume
    global_step = 1
    run_name = config["run_name"]
    project_name = config["project_name"]
    save_path = Path(config["save_path"]) / project_name / run_name
    marker_file = save_path / "latest_checkpointed_iteration.txt"
    if marker_file.exists():
        with open(marker_file) as f:
            step = f.read().strip()
        if step:
            global_step = int(step) + 1
            resume_path = save_path / f"global_step_{step}"
            print(f"resuming training from {resume_path}")
            actor.load_checkpoint(str(resume_path))
    else:
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.touch()

    # make sure workers are ready, use no_op as a barrier
    ray.get(rollout.no_op() + actor.no_op() + ref.no_op())

    # send initial params to rollout (distributed broadcast)
    sync_weights()

    val_task_manager = DistributedTaskManager(
        task_fn=lambda item: multimodal_chat_task(
            item,
            config=task_config,
            tokenizer=tokenizer,
            gen_fn=partial(
                rollout.dispatch_rank(hash(str(item)) % rollout.world_size).generate_with_ids,
                sampling_params=val_config.get("sampling_params", {}),
            ),
        ),
        max_queue_size=val_concurrency * 10,
        max_buffer_size=val_concurrency * 10,
        buffer_group_size=1,
        num_workers=val_concurrency,
        event_loop=event_loop,
    )
    val_task_manager.start()

    validating = False
    def load_val_data():
        val_dataloader = repeat(DataLoader(ConcatDataset(val_datasets), batch_size=None, shuffle=True), val_config["n"])
        count = 0
        for item in val_dataloader:
            asyncio.run_coroutine_threadsafe(val_task_manager.put(item), event_loop)
            count += 1

    if config.get("val_before_train", False):
        load_val_data()
        validating = True

    # start training
    max_steps = config["max_steps"]
    save_interval = config.get("save_interval", 50)
    val_interval = config.get("val_interval", 25)
    train_dataloader_generator = torch.Generator()
    train_dataloader_generator.manual_seed(42)
    dataloader = iter(interleave(*[cycle(repeat(DataLoader(
        ds,
        batch_size=None,
        shuffle=True,
        generator=train_dataloader_generator,
    ), n)) for ds in train_datasets]))
    wandb.init(project=project_name, name=run_name, group=run_name, config=config)

    timer = Timer()
    first_step = True
    while global_step < max_steps:
        timer.step_start()
        metrics = {}

        # data
        with timer.time("prepare_data"):
            for _ in range(train_task_manager.queue_maxsize - train_task_manager.queue_size):
                item = next(dataloader)
                train_task_manager.put_nowait(item)
        with timer.time("gen"):
            _lcm = math.lcm(n, train_world)
            _raw = real_bsz * (1.7 if first_step else 1)
            _need = max(_lcm, int(math.ceil(_raw / _lcm) * _lcm))
            print(f"[trainer] requesting {_need} items "
                  f"(real_bsz={real_bsz}, n={n}, train_world={train_world}, "
                  f"first_step={first_step})...", flush=True)
            data = asyncio.run_coroutine_threadsafe(
                train_task_manager.get(_need, n),
                event_loop,
            ).result()
            print(f"[trainer] got {len(data)} items -> starting training step "
                  f"{global_step}", flush=True)
            first_step = False

        # === multimodal: fill mrope position_ids ===
        with timer.time("positions"):
            from agentrl.trainer.components.position_fill_mm import fill_position_ids
            img_items = [it for it in data
                         if (it.get("multi_modal_inputs") or {}).get("image_grid_thw") is not None]
            if img_items:
                ids_list = [it["input_ids"] for it in img_items]
                grid_list = [it["multi_modal_inputs"]["image_grid_thw"] for it in img_items]
                print(f"[TRAINER] calling compute_positions, img_items={len(img_items)}", flush=True)
                positions = ray.get(actor.compute_positions(ids_list, grid_list))[0]
                print(f"[TRAINER] compute_positions returned", flush=True)
                for it, pos in zip(img_items, positions):
                    it["position_ids"] = pos
            fill_position_ids(data, get_rope_index_fn=None)

        # advantage
        with timer.time("adv"):
            adv = compute_advantage(data, **config.get("advantage", {}))
            for item, adv_item in zip(data, adv):
                item.update(adv_item)

        # ref
        with timer.time("ref"):
            loss_config = config.get("loss", {})
            print(f"[TRAINER] calling ref.forward_backward, data items={len(data)}", flush=True)
            log_probs = ray.get(ref.forward_backward(
                data, partial(log_prob_loss, config=loss_config), forward_only=True, unpack=True,
            ))[0]
            print(f"[TRAINER] ref.forward_backward returned", flush=True)
            for item, log_prob in zip(data, log_probs["log_prob"]):
                item["ref_log_prob"] = log_prob

        # actor
        with timer.time("update_actor"):
            for item in data:
                if actor_config["loss_reduce_mode"] == "seq-mean":
                    item["loss_weight"] = 1
                elif actor_config["loss_reduce_mode"] == "token-mean":
                    item["loss_weight"] = item["loss_tokens"]
                else:
                    raise NotImplementedError(f"unknown reduce mode {actor_config['reduce_mode']}")
            train_metrics = ray.get(actor.forward_backward(
                data, partial(ppo_loss, config=loss_config), unpack=True,
            ))[0]
            grad_norm = ray.get(actor.step())[0]
            train_metrics["grad_norm"] = grad_norm
            append_with_prefix(metrics, "actor/", reduce_dict(train_metrics))

        data_metrics = gather_metrics(data)
        append_with_prefix(metrics, "rl/", data_metrics)

        if validating:
            val_metrics = collect_val_metrics(val_task_manager, event_loop)
            append_with_prefix(metrics, "val/", val_metrics)
            validating = False

        # sync params (distributed NCCL broadcast: trainer rank0 -> SGLang)
        with timer.time("sync_params"):
            sync_weights()

        if global_step % val_interval == 0:
            load_val_data()
            validating = True

        if global_step % save_interval == 0:
            actor.save_checkpoint(str(save_path / f"global_step_{global_step}"))
            with open(save_path / "latest_checkpointed_iteration.txt", "w") as f:
                f.write(str(global_step))

        timing_metrics = timer.step_end()
        global_metrics = calc_data_metrics(data)
        global_metrics["throughput"] = global_metrics["total_seq_len"] / timing_metrics["step"]
        global_metrics["throughput_per_device"] = global_metrics["throughput"] / total_gpus
        append_with_prefix(metrics, "timings/", timing_metrics)
        append_with_prefix(metrics, "global/", global_metrics)

        print(f"Step {global_step}")
        pretty_print_metrics(metrics)
        wandb.log(metrics, step=global_step)
        global_step += 1
    print("training completed.")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Async Multimodal Trainer")
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    agent_repo_dir = config.get("task", {}).get("agent_repo_dir")
    if agent_repo_dir:
        import sys
        if agent_repo_dir not in sys.path:
            sys.path.insert(0, agent_repo_dir)
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            agent_repo_dir + (":" + existing if existing else ""))

    ray.init(runtime_env={"env_vars": dict(os.environ)})

    main(config)