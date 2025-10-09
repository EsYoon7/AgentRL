import torch

from .core_algos import agg_loss, kl_penalty, compute_policy_loss
from ..utils.torch_functional import logprobs_from_logits, entropy_from_logits


def ppo_loss(model_inputs, output, config):
    loss_mask = model_inputs.pop("loss_mask")
    old_log_prob = model_inputs.pop("rollout_log_prob")
    advantages = model_inputs.pop("advantages")
    input_ids_rolled = torch.roll(model_inputs["input_ids"], shifts=-1, dims=1)

    assert (loss_mask[:, 0] == 0).all(), "The first token should not have loss on it."

    logits = output["logits"] / config["temperature"]
    assert input_ids_rolled.shape[:2] == logits.shape[:2], f"{input_ids_rolled.shape} != {logits.shape}"
    log_prob = logprobs_from_logits(
        logits=logits,
        labels=input_ids_rolled,
        inplace_backward=False,
    )
    entropy = entropy_from_logits(
        logits=logits,
    )
    log_prob = torch.roll(log_prob, shifts=1, dims=1)
    entropy = torch.roll(entropy, shifts=1, dims=1)

    clip_ratio = config.get("clip_ratio", 0.2)
    clip_ratio_low = config.get("clip_ratio_low", clip_ratio)
    clip_ratio_high = config.get("clip_ratio_high", clip_ratio)
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    entropy_coef = config["entropy_coef"]
    loss_agg_mode = config.get("loss_agg_mode", "token-mean")

    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        loss_mask=loss_mask,
        cliprange=clip_ratio,
        cliprange_low=clip_ratio_low,
        cliprange_high=clip_ratio_high,
        clip_ratio_c=clip_ratio_c,
        loss_agg_mode=loss_agg_mode,
    )

    entropy_loss = agg_loss(
        loss_mat=entropy, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode
    )
    if entropy_coef != 0:
        # compute policy loss
        policy_loss = pg_loss - entropy_loss * entropy_coef
    else:
        policy_loss = pg_loss

    metric = {
        "pg_loss": pg_loss.detach().item(),
        "entropy_loss": entropy_loss.detach().item(),
        "pg_clipfrac": pg_clipfrac.detach().item(),
        "ppo_kl": ppo_kl.detach().item(),
        "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    if config.get("use_kl_loss", False):
        ref_log_prob = model_inputs["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(
            logprob=log_prob,
            ref_logprob=ref_log_prob,
            kl_penalty=config.get("kl_loss_type", "low_var_kl"),
        )
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode
        )

        kl_loss_coef = config["kl_loss_coef"]
        policy_loss = policy_loss + kl_loss * kl_loss_coef
        metric["kl_loss"] = kl_loss.detach().item()
        metric["kl_coef"] = kl_loss_coef

    return policy_loss, metric


def log_prob_loss(model_inputs, output, config):
    input_ids_rolled = torch.roll(
        model_inputs["input_ids"], shifts=-1, dims=1,
    )
    logits = output["logits"] / config["temperature"]
    assert input_ids_rolled.shape[:2] == logits.shape[:2], f"{input_ids_rolled.shape} != {logits.shape}"
    log_prob = logprobs_from_logits(
        logits=logits,
        labels=input_ids_rolled,
        inplace_backward=False,
    )
    log_prob = torch.roll(log_prob, shifts=1, dims=1)
    assert log_prob.shape == model_inputs["input_ids"].shape, f"{log_prob.shape} != {model_inputs['input_ids'].shape}"
    return torch.tensor(1.0), {"log_prob": log_prob}


def cross_entropy_loss(model_inputs, output):
    labels = model_inputs["input_ids"][:, 1:].clone()
    loss_mask = model_inputs["loss_mask"][:, 1:]
    labels[~loss_mask.bool()] = -100
    logits = output["logits"][:, :-1]
    ce_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )
    return ce_loss, {"loss": ce_loss.detach().item()}
