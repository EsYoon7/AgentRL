from collections import defaultdict

import numpy as np
import torch

from ..utils import to_device


def calc_batch_rl_metrics(items):
    grouped_scores = defaultdict(list)

    # Iterate over the batch to collect scores for each unique id
    for item in items:
        group_id = item["group_id"]
        score = item["reward"]
        grouped_scores[group_id].append(score)

    grouped_scores = list(grouped_scores.values())
    return {
        "average_reward": np.mean([np.mean(scores) for scores in grouped_scores]),
        "BoN_reward": np.mean([max(scores) for scores in grouped_scores]),
        "average_margin": np.mean([max(scores) - sum(scores) / len(scores) for scores in grouped_scores]),
        "num_uids": len(grouped_scores),
        "effective_uids": np.mean([len(set(scores)) > 1 for scores in grouped_scores]),
        "all_1_uids": np.mean([len(set(scores)) == 1 and scores[0] == 1 for scores in grouped_scores]),
        "all_leq0_uids": np.mean([len(set(scores)) == 1 and scores[0] <= 0 for scores in grouped_scores]),
        "average_repeat": np.mean([len(scores) for scores in grouped_scores]),
    }

def calc_adv_metrics(items):
    all_adv = []
    for item in items:
        adv = to_device(item["advantages"], "cpu")
        loss_mask = to_device(item["loss_mask"], "cpu")
        effective_adv = torch.masked_select(adv, loss_mask.bool())
        all_adv.append(effective_adv)
    all_adv = torch.cat(all_adv, dim=0)
    return {
        "adv/mean": all_adv.mean().item(),
        "adv/std": all_adv.std().item(),
        "adv/max": all_adv.max().item(),
        "adv/min": all_adv.min().item(),
    }

def calc_metrics(items):
    metric_by_keys = defaultdict(list)
    for item in items:
        for key, value in item.get("metrics", {}).items():
            metric_by_keys[key].append(value)

    return {k: np.mean(v) for k, v in metric_by_keys.items()}


def calc_data_metrics(items):
    seq_lens = [item["seq_len"] for item in items]
    return {
        "batch_size": len(items),
        "total_seq_len": sum(seq_lens),
        "seq_len/max": max(seq_lens),
        "seq_len/min": min(seq_lens),
        "seq_len/std": np.std(seq_lens),
        "seq_len/avg": sum(seq_lens) / len(seq_lens),
    }
