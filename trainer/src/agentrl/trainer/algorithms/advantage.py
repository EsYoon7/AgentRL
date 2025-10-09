from enum import Enum

from . import core_algos
from ..utils import to_device


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator."""

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"


def compute_advantage(
    data: list[dict],
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    norm_adv_by_std_in_grpo: bool = True,
) -> list[dict]:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data: List of dicts containing model outputs and inputs. Each dict must contain:
            - token_level_rewards: torch.Tensor of shape (response_length,)
            - loss_mask: torch.Tensor of shape (response_length,)
            - values: torch.Tensor of shape (response_length,) if using GAE
            - reward_baselines: torch.Tensor of shape () if using REMAX
            - uid: Any hashable type, used for grouping samples
        adv_estimator: The advantage estimator to use
        gamma: Discount factor for future rewards
        lam: Lambda parameter for GAE
        norm_adv_by_std_in_grpo: Whether to normalize advantages by standard deviation in GRPO

    Returns:
        List of dicts with the same length as input, each containing:
            - advantages: torch.Tensor of shape (response_length,)
            - returns: torch.Tensor of shape (response_length,)
    """
    from collections import defaultdict

    # Algorithms that require group computation
    group_algos = [
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_PASSK,
        AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        AdvantageEstimator.RLOO,
    ]

    ret: list[dict | None] = [None] * len(data)

    if adv_estimator in group_algos:
        # group by uid
        group_dict: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for idx, item in enumerate(data):
            group_dict[item["group_id"]].append((idx, item))
        for group in group_dict.values():
            idxs, group_items = zip(*group)
            token_level_rewards = [to_device(item["token_level_rewards"], "cpu") for item in group_items]
            loss_mask = [to_device(item["loss_mask"], "cpu") for item in group_items]
            if adv_estimator == AdvantageEstimator.GRPO:
                advantages, returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, loss_mask, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
                )
            elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
                advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
                    token_level_rewards, loss_mask, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
                )
            elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
                advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
                    token_level_rewards, loss_mask
                )
            elif adv_estimator == AdvantageEstimator.RLOO:
                advantages, returns = core_algos.compute_rloo_outcome_advantage(
                    token_level_rewards, loss_mask
                )
            else:
                raise NotImplementedError
            for idx, adv, ret_val in zip(idxs, advantages, returns):
                ret[idx] = {
                    "advantages": adv,
                    "returns": ret_val,
                }
        return ret

    # algorithms that can be processed per item
    for item in data:
        if adv_estimator == AdvantageEstimator.GAE:
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=item["token_level_rewards"],
                values=item["values"],
                loss_mask=item["loss_mask"],
                gamma=gamma,
                lam=lam,
            )
        elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
            advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
                token_level_rewards=item["token_level_rewards"],
                loss_mask=item["loss_mask"],
                gamma=gamma,
            )
        elif adv_estimator == AdvantageEstimator.REMAX:
            advantages, returns = core_algos.compute_remax_outcome_advantage(
                token_level_rewards=item["token_level_rewards"],
                reward_baselines=item["reward_baselines"],
                loss_mask=item["loss_mask"],
            )
        else:
            raise NotImplementedError
        ret.append({
            "advantages": advantages,
            "returns": returns,
        })
    return ret
