import argparse
import asyncio
import os
import threading
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import cycle
from pathlib import Path
import random

import ray
import torch
import wandb
import yaml
from ray.util import placement_group
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoProcessor

from agentrl.trainer.agentic.data_provider import get_agentic_datasets
from agentrl.trainer.algorithms.advantage import compute_advantage
from agentrl.trainer.algorithms.loss_funcs import log_prob_loss, ppo_loss
from agentrl.trainer.algorithms.metrics import calc_metrics, calc_batch_rl_metrics, calc_data_metrics, calc_adv_metrics
from agentrl.trainer.components.nccl_tensor_comm import NCCLTensorSender, NCCLTensorReceiver
from agentrl.trainer.components.task_manager import openai_chat_task, DistributedTaskManager
from agentrl.trainer.components.timer import Timer
from agentrl.trainer.utils import append_with_prefix, reduce_dict, pretty_print_metrics, repeat, interleave, to_device
from agentrl.trainer.workers.async_sglang_worker import AsyncSglangWorker
from agentrl.trainer.workers.collective_handle import spawn
from agentrl.trainer.workers.fsdp_worker import FSDPWorker


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


def adv_norm(items):
    all_adv = []
    for item in items:
        adv = to_device(item["advantages"], "cpu")
        loss_mask = to_device(item["loss_mask"], "cpu")
        effective_adv = torch.masked_select(adv, loss_mask.bool())
        all_adv.append(effective_adv)
    all_adv = torch.cat(all_adv, dim=0)
    mean = all_adv.mean()
    std = all_adv.std()
    for item in items:
        item["advantages"] = (item["advantages"] - mean) / (std + 1e-6)


def main(config):
    # spawn workers
    rollout_config = config.get("rollout", {})
    actor_config = config.get("actor", {})
    ref_config = deepcopy(actor_config)
    ref_config.update(config.get("ref", {}))
    task_config = config.get("task", {})

    total_gpus = int(ray.cluster_resources()["GPU"])
    rollout_gpus = int(config.get("rollout_ratio", 0.5) * total_gpus)
    stale_ratio = config.get("rollout_stale_ratio", 0.0)
    rollout_stale_gpus = int(stale_ratio * rollout_gpus)
    actor_gpus = total_gpus - rollout_gpus
    print(f"Total GPUs: {total_gpus}, Rollout GPUs: {rollout_gpus - rollout_stale_gpus} Rollout Stale GPUs: {rollout_stale_gpus}, Actor GPUs: {actor_gpus}")

    rollout_tp = rollout_config.get("tp_size", 1)
    rollout_placement = placement_group([{"CPU": 1, "GPU": rollout_tp}] * ((rollout_gpus - rollout_stale_gpus) // rollout_tp))
    rollout_stale_placement = placement_group([{"CPU": 1, "GPU": rollout_tp}] * (rollout_stale_gpus // rollout_tp)) if rollout_stale_gpus > 0 else None
    actor_ref_placement = placement_group([{"CPU": 1, "GPU": 1}] * actor_gpus)

    rollout = spawn(AsyncSglangWorker, rollout_placement, num_gpus=rollout_tp)(rollout_config)
    rollout_stale = spawn(AsyncSglangWorker, rollout_stale_placement, num_gpus=rollout_tp)(rollout_config) if rollout_stale_placement else None
    actor = spawn(FSDPWorker, actor_ref_placement, num_gpus=0.5)(actor_config)
    ref = spawn(FSDPWorker, actor_ref_placement, num_gpus=0.5)(ref_config)

    # initialize workers
    streamer_ip, streamer_port = ray.get(actor.dispatch_rank0().get_addr_and_port())
    streamer_world_size = 1 + len(rollout.workers)
    streamer_args = (streamer_ip, streamer_port, streamer_world_size)

    rollout.build_engine(config["model_path"])
    rollout.register_plugin("param_receiver", NCCLTensorReceiver, *streamer_args, offset=1)

    actor.build_model(config["model_path"])
    actor.build_optimizer()
    actor.build_checkpoint_manager()
    actor.register_plugin("param_sender", NCCLTensorSender, *streamer_args)

    if rollout_stale:
        streamer2_ip, streamer2_port = ray.get(actor.dispatch_rank0().get_addr_and_port())
        streamer2_world_size = 1 + len(rollout_stale.workers)
        streamer2_args = (streamer2_ip, streamer2_port, streamer2_world_size)
        rollout_stale.build_engine(config["model_path"])
        rollout_stale.register_plugin("stale_receiver", NCCLTensorReceiver, *streamer2_args, offset=1)
        actor.register_plugin("stale_sender", NCCLTensorSender, *streamer2_args)

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

    def cross_sampler(item, **kwargs):
        if rollout_stale is None:
            return rollout.dispatch_rank(hash(str(item)) % rollout.world_size).generate(**kwargs)
        else:
            if random.random() < (1 - stale_ratio):
                return rollout.dispatch_rank(hash(str(item)) % rollout.world_size).generate(**kwargs)
            else:
                return rollout_stale.dispatch_rank(hash(str(item)) % rollout_stale.world_size).generate(**kwargs)

    # Create training task manager
    train_task_manager = DistributedTaskManager(
        task_fn=lambda item: openai_chat_task(
            item,
            config=task_config,
            tokenizer=tokenizer,
            gen_fn=partial(
                cross_sampler,
                item=item,
                sampling_params=train_config.get("sampling_params", {}),
            ),
        ),
        max_queue_size=real_bsz * 2,
        max_buffer_size=real_bsz * 2,
        buffer_group_size=n,
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

    # send params to rollout
    actor.call_plugin("param_sender", "send", float(config.get("bucket_size", 1e9)))
    rollout.async_call_plugin("param_receiver", "async_receive")
    if rollout_stale:
        actor.call_plugin("stale_sender", "send", float(config.get("bucket_size", 1e9)))
        rollout_stale.async_call_plugin("stale_receiver", "async_receive")

    # Create validation task manager
    val_task_manager = DistributedTaskManager(
        task_fn=lambda item: openai_chat_task(
            item,
            config=task_config,
            tokenizer=tokenizer,
            gen_fn=partial(
                rollout.dispatch_rank(hash(str(item)) % rollout.world_size).generate,
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

    # Helper to load validation data
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
            data = asyncio.run_coroutine_threadsafe(
                train_task_manager.get(real_bsz * (1.7 if first_step else 1), n),
                event_loop,
            ).result()
            first_step = False

        # advantage
        with timer.time("adv"):
            adv = compute_advantage(data, **config.get("advantage", {}))
            for item, adv_item in zip(data, adv):
                item.update(adv_item)

        # ref
        with timer.time("ref"):
            loss_config = config.get("loss", {})
            log_probs = ray.get(ref.forward_backward(
                data, partial(log_prob_loss, config=loss_config), forward_only=True, unpack=True,
            ))[0]
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

        if config.get("task_adv_norm"):
            by_source = defaultdict(list)
            for item in data:
                by_source[item["data_source"]].append(item)
            for items in by_source.values():
                adv_norm(items)

        data_metrics = gather_metrics(data)
        append_with_prefix(metrics, "rl/", data_metrics)

        # collect val tasks
        if validating:
            val_metrics = collect_val_metrics(val_task_manager, event_loop)
            append_with_prefix(metrics, "val/", val_metrics)
            validating = False

        # sync params
        with timer.time("sync_params"):
            r = []
            r += actor.call_plugin("param_sender", "send", float(config.get("bucket_size", 1e9)))
            r += rollout.async_call_plugin("param_receiver", "async_receive")
            if rollout_stale and global_step % int(config["stale_step"]) == 0:
                r += actor.call_plugin("stale_sender", "send", float(config.get("bucket_size", 1e9)))
                r += rollout_stale.async_call_plugin("stale_receiver", "async_receive")
            ray.get(r)

        # issue val tasks
        if global_step % val_interval == 0:
            load_val_data()
            validating = True

        if global_step % save_interval == 0:
            # save checkpoint
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
    parser = argparse.ArgumentParser(description="Run Async Trainer")
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ray.init(runtime_env={"env_vars": dict(os.environ)})

    main(config)
