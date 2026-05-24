import contextlib
import math
import os
import typing
from datetime import timedelta
from typing import Callable

import binpacking
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.distributed import init_device_mesh, DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from .abstract import AbstractTrainWorker, LossFnType
from ..utils import append_to_dict, to_device, to_plasma, clean_cuda

if typing.TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from ..checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]
    elif isinstance(fsdp_transformer_layer_cls_to_wrap, (set, frozenset, tuple)):
        # Some HF configs (e.g. Qwen3.5) expose `_no_split_modules` as a set;
        # normalize to list so indexing/iteration below works regardless of source.
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)

    assert (
        len(fsdp_transformer_layer_cls_to_wrap) > 0
        and fsdp_transformer_layer_cls_to_wrap[0] is not None
    )

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs)
    fully_shard(
        model, **fsdp_kwargs
    )  # fsdp2 will not reshard_after_forward for root module


def pack(data: list[dict], max_tokens, multiple_of: int = 1) -> list[list[dict]]:
    seq_lens = {i: d["seq_len"] for i, d in enumerate(data)}
    bins = binpacking.to_constant_volume(seq_lens, max_tokens)
    num_bins = math.ceil(len(bins) / multiple_of) * multiple_of
    bins = binpacking.to_constant_bin_number(seq_lens, num_bins)
    packed_data = [[data[i] for i in b] for b in bins]
    return packed_data


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[list[dict]], collator: Callable[[list[dict]], dict]):
        self.data = data
        self.collator = collator

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[int, dict]:
        return idx, self.collator(self.data[idx])


class FSDPWorker(AbstractTrainWorker):
    device_mesh: DeviceMesh
    model: FSDPModule | nn.Module
    optimizer: Optimizer
    processor: "PreTrainedTokenizerBase"
    checkpoint_manager: "FSDPCheckpointManager"

    def __init__(self, config):
        super().__init__()
        self.config = config

    def init_distributed(self, addr, port):
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{addr}:{port}",
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(minutes=60),
        )

        # build device mesh for FSDP
        world_size = dist.get_world_size()
        fsdp_size = self.config.get("fsdp_size", -1)
        if fsdp_size == -1:
            fsdp_size = world_size
        dp_size = world_size // fsdp_size
        self.device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(dp_size, fsdp_size),
            mesh_dim_names=("dp", "fsdp"),
        )

    def build_model(self, path):
        from transformers import AutoModelForCausalLM, AutoProcessor
        fsdp_config = self.config.get("fsdp_config", {})
        # load model
        torch_dtype = getattr(torch, self.config["torch_dtype"])
        model_class_name = self.config.get("model_class", "causal_lm")
        if model_class_name == "causal_lm":
            auto_cls = AutoModelForCausalLM
        elif model_class_name == "image_text_to_text":
            from transformers import AutoModelForImageTextToText
            auto_cls = AutoModelForImageTextToText
        else:
            raise ValueError(f"unknown model_class {model_class_name!r}; expected 'causal_lm' or 'image_text_to_text'")
        self.model = auto_cls.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=torch_dtype,
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
        )
        self.model.to(torch_dtype)
        if self.config.get("enable_gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        # load processor
        self.processor = AutoProcessor.from_pretrained(path)

        # apply FSDP
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
        )
        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mp_policy,
            "reshard_after_forward": self.config.get("reshard_after_forward", False),
        }
        apply_fsdp2(self.model, fsdp_kwargs, fsdp_config)
        assert isinstance(self.model, FSDPModule)

    def build_optimizer(self):
        optim_config = self.config.get("optim", {})
        assert self.model is not None, "Model must be built before optimizer"
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            **optim_config,
        )

    def build_checkpoint_manager(self):
        from ..checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

        assert self.model is not None, "Model must be built before checkpoint manager"
        assert (
            self.optimizer is not None
        ), "Optimizer must be built before checkpoint manager"
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=None,
            processing_class=self.processor,
            checkpoint_contents=self.config.get("checkpoint_contents"),
        )

    def param_generator(self):
        for k, v in self.model.named_parameters():
            if isinstance(v, DTensor):
                yield k, v.full_tensor().to(torch.bfloat16)
            else:
                yield k, v

    def collate(self, batch: list[dict], packing=True) -> dict:
        """
        Collate function to prepare model inputs from a batch of data.
        For Tensors: concatenate along the sequence dimension (dim=1) if packing is True.
        For Tensors inside `multi_modal_inputs`: stack them along a new dimension (dim=1).
        For other types: collect them into lists.
        :param batch: List of dictionaries, where each dictionary represents a data item.
        :param packing: If True, concatenate Tensors along the sequence dimension.
        :return: A dictionary containing model inputs ready for the model.
        """
        model_text_inputs = {}
        model_multi_modal_inputs = {}
        for item in batch:
            item = to_device(item)
            assert item["input_ids"].shape[1] == item["seq_len"], f"{item['input_ids']=}, {item['seq_len']=}"
            for key, value in item.items():
                if key == "multi_modal_inputs":
                    for k, v in value.items():
                        if k not in model_multi_modal_inputs:
                            model_multi_modal_inputs[k] = []
                        model_multi_modal_inputs[k].append(v)
                else:
                    if key not in model_text_inputs:
                        model_text_inputs[key] = []
                    model_text_inputs[key].append(value)

        model_inputs = {}
        indices = torch.cumsum(torch.tensor(model_text_inputs["seq_len"]), dim=0)
        assert not packing or "position_ids" in model_text_inputs, "position_ids must be passed if enabling packing."
        for k, v in model_text_inputs.items():
            if isinstance(v[0], torch.Tensor):
                if packing:
                    model_inputs[k] = torch.cat(v, dim=1).cuda()
                else:
                    raise NotImplementedError
            else:
                model_inputs[k] = v
        for k, v in model_multi_modal_inputs.items():
            if isinstance(v[0], torch.Tensor):
                model_inputs[k] = torch.stack(v, dim=1).cuda()
            else:
                model_inputs[k] = v
        model_inputs["indices"] = indices

        return model_inputs

    def _create_distributed_loader(self, data: list[list[dict]]):
        ds = ListDataset(data, self.collate)
        sampler = DistributedSampler(ds, shuffle=False)
        return DataLoader(
            dataset=ds,
            batch_size=None,
            sampler=sampler,
            num_workers=0,
        )

    @clean_cuda
    def forward_backward(
        self,
        mini_batch: list[dict],
        loss_fn: LossFnType,
        forward_only: bool = False,
        unpack: bool = False,
    ):
        """
        Forward and backward pass for the model.
        if `unpack` is False:
            Directly calculate loss over packed data. returning metrics sorted by micro_batch index.
            This is useful for training.
        If `unpack` is True:
            Unpack the micro-batch into individual samples, calculate loss for each sample,
            and return metrics sorted by sample index. This is useful for calculating log_probs.
        """
        assert self.model is not None, "Model must be built before forward_backward"
        assert forward_only or self.optimizer is not None, "Optimizer must be built before forward_backward"
        assert isinstance(mini_batch, list)

        if forward_only:
            self.model.eval()
            cm = torch.no_grad()
        else:
            self.model.train()
            cm = contextlib.nullcontext()

        full_weight = 0
        for i, item in enumerate(mini_batch):
            item["sample_idx"] = i
            if "loss_weight" not in item:
                item["loss_weight"] = 1.0
            full_weight += item["loss_weight"]
        full_weight /= self.device_mesh.size()
        micro_batches = pack(
            mini_batch,
            max_tokens=self.config["max_tokens_per_micro_batch"],
            multiple_of=self.device_mesh.size(),
        )
        dataloader = self._create_distributed_loader(micro_batches)

        metrics = []
        total_batches = len(dataloader)
        for batch_idx, inputs in dataloader:
            self.model.set_is_last_backward(batch_idx == total_batches)
            model_inputs = {k: v for k, v in inputs.items() if k in self.config["model_input_keys"]}
            batch_weight = sum(inputs["loss_weight"])

            with cm:
                output = self.model(**model_inputs)

            if unpack:
                indices = inputs.pop("indices")
                last_idx = 0
                all_loss = []
                for i, index in enumerate(indices):
                    output_item = {
                        k: v[:, last_idx: index] if isinstance(v, torch.Tensor) else v
                        for k, v in output.items()
                    }
                    sample_idx = inputs["sample_idx"][i]
                    item = micro_batches[batch_idx][i]
                    assert mini_batch[sample_idx]["uid"] == item["uid"]
                    loss, metric = loss_fn(to_device(item), output_item)
                    metrics.append((sample_idx, metric))
                    all_loss.append(loss * item["loss_weight"] / batch_weight)
                    last_idx = index
                loss = torch.stack(all_loss, dim=0).sum()
            else:
                loss, metric = loss_fn(inputs, output)
                metrics.append((batch_idx, metric))

            if not forward_only:
                loss = loss * batch_weight / full_weight
                loss.backward()

        torch.cuda.empty_cache()
        # gather metrics from all ranks
        all_metrics: list[None | list[tuple[int, dict]]] = [None for _ in range(self.device_mesh.size())]
        dist.all_gather_object(all_metrics, to_plasma(metrics))
        flattened_metrics = [m for sublist in all_metrics for m in sublist]
        flattened_metrics.sort(key=lambda x: x[0])  # sort by idx

        metrics = {}
        for _, metric in flattened_metrics:
            append_to_dict(metrics, metric)

        return to_plasma(metrics)

    @clean_cuda
    def step(self):
        assert self.optimizer is not None, "Optimizer must be built before step"
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.get("grad_clip", 1.0),
        )
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def save_checkpoint(self, path: str):
        assert self.checkpoint_manager is not None, "Checkpoint manager must be built before saving checkpoint"
        self.checkpoint_manager.save_checkpoint(path, max_ckpt_to_keep=self.config.get("max_ckpt_to_keep"))

    def load_checkpoint(self, path: str):
        assert self.checkpoint_manager is not None, "Checkpoint manager must be built before loading checkpoint"
        self.checkpoint_manager.load_checkpoint(path)
