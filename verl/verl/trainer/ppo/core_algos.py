# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from math import comb


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
    
def compute_qae_advantage(token_level_rewards, response_mask, index, quantile_K=0.4, eps=1e-8):
    """
    Implements Quantile Advantage Estimation (QAE) from arXiv:2509.22611v1
    - Works with binary outcome rewards R ∈ {0, 1}
    - Computes quantile baseline b_K per group (query)
    - Standardizes by empirical std(p(1-p)) + eps
    """
    import torch
    from collections import defaultdict

    rewards = token_level_rewards.sum(dim=-1)  # per response total reward
    device = rewards.device
    group_map = defaultdict(list)
    for i, uid in enumerate(index):
        group_map[uid].append(i)

    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    for _, idxs in group_map.items():
        idxs_t = torch.tensor(idxs, device=device)
        Rs = rewards[idxs_t]
        p = Rs.mean().item()
        std = (p * (1 - p)) ** 0.5
        bK = 0.0 if p <= 1 - quantile_K else 1.0
        A = (Rs - bK) / (std + eps)
        advantages[idxs_t] = A
        returns[idxs_t] = Rs

    # reshape to (batch_size, seq_len)
    advantages = advantages.unsqueeze(1).expand_as(token_level_rewards) * response_mask
    returns = returns.unsqueeze(1).expand_as(token_level_rewards) * response_mask
    return advantages, returns
#75%quantile
def compute_quantile_advantage(token_level_rewards, response_mask, index, quantile_K, eps=1e-8):
    """
    Implements Quantile Advantage Estimation (QAE) from arXiv:2509.22611v1
    - Works with binary outcome rewards R ∈ {0, 1}
    - Computes quantile baseline b_K per group (query)
    - Standardizes by empirical std(p(1-p)) + eps
    """
    import torch
    from collections import defaultdict

    rewards = token_level_rewards.sum(dim=-1)  # per response total reward
    device = rewards.device
    group_map = defaultdict(list)
    for i, uid in enumerate(index):
        group_map[uid].append(i)

    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    for _, idxs in group_map.items():
        idxs_t = torch.tensor(idxs, device=device)
        Rs = rewards[idxs_t]
        p = Rs.mean().item()
        std = (p * (1 - p)) ** 0.5
        bK = torch.quantile(Rs.float(), quantile_K).item()
        A = (Rs - bK) / (std + eps)
        advantages[idxs_t] = A
        returns[idxs_t] = Rs

    # reshape to (batch_size, seq_len)
    advantages = advantages.unsqueeze(1).expand_as(token_level_rewards) * response_mask
    returns = returns.unsqueeze(1).expand_as(token_level_rewards) * response_mask
    return advantages, returns

def calc_adv(val, k):
    c = len(np.where(val==1)[0])
    n = len(val)
    if n < k or comb(n, k) == 0:
        return np.zeros_like(val, dtype=float)
    rho = 1 - comb(n-c, k) / comb(n, k)
    sigma = np.sqrt(rho * (1 - rho))
    adv_p = (1 - rho) / (sigma + 1e-6)
    adv_n = (1 - rho - comb(n-c-1, k-1)/comb(n-1,k-1)) / (sigma + 1e-6)
    new_val = np.where(val==1, adv_p, val)
    new_val = np.where(new_val==0, adv_n, new_val)
    return new_val
def compute_passktraining_advantage(token_level_rewards, response_mask, index, K):
    scores = token_level_rewards.sum(dim=-1)
    
    id2score = defaultdict(list)
    uid2sid = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i].detach().item())
            uid2sid[index[i]].append(i)
        for uid in id2score.keys():
            reward = np.array(id2score[uid])
            adv = calc_adv(reward, K)
#            print(uid2sid[uid])
            for i in range(len(uid2sid[uid])):
                scores[uid2sid[uid][i]] = adv[i]

    scores = scores.unsqueeze(-1) * response_mask
    
    return scores, scores

def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac

#Huber loss
def compute_IQN_quantile_value_loss(
    vpreds: torch.FloatTensor,
    target_quantiles: torch.FloatTensor,
    response_mask: torch.FloatTensor,
    kappa: float = 1.0,
    taus: torch.FloatTensor = None,
    target_taus: torch.FloatTensor = None,
):
    assert vpreds.dim() == 3, f"Expected vpreds (bs, T, K), got {vpreds.shape}"
    assert target_quantiles.dim() == 3, f"Expected target_quantiles (bs, T, K'), got {target_quantiles.shape}"
    bs, t, k = vpreds.shape
    b2, t2, k_target = target_quantiles.shape
    assert bs == b2 and t == t2, f"Shape mismatch between vpreds {vpreds.shape} and target_quantiles {target_quantiles.shape}"

    if taus is None:
        taus = torch.rand(bs, t, k, device=vpreds.device, dtype=vpreds.dtype)
    else:
        assert taus.shape == (bs, t, k), f"taus should have shape (bs, T, K), got {taus.shape}"
    if target_taus is not None:
        assert target_taus.shape == (bs, t, k_target), f"target_taus should have shape (bs, T, K'), got {target_taus.shape}"

    # Diff across both tau and tau' (bs, T, K, K')
    target_q = target_quantiles.detach().unsqueeze(-2)
    pred_q = vpreds.unsqueeze(-1)
    diff = target_q - pred_q
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= kappa, 0.5 * diff.pow(2), kappa * (abs_diff - 0.5 * kappa))

    tau = taus.unsqueeze(-1)  # (bs, T, K, 1)
    quantile_loss = torch.abs(tau - (diff < 0).float()) * huber
    # mean over target quantiles K'
    quantile_loss = quantile_loss.mean(dim=-1)  # (bs, T, K)

    loss_mask = response_mask.unsqueeze(-1)  # (bs, T, 1)
    loss = verl_F.masked_mean(quantile_loss, loss_mask)
    return loss

#Huber loss
def compute_quantile_value_loss(
    vpreds: torch.FloatTensor,
    returns: torch.FloatTensor,
    response_mask: torch.FloatTensor,
    num_quantiles: int,
    tau_mode: str = "iqn",
    kappa: float = 1.0,
    taus: torch.FloatTensor = None,
):
    assert vpreds.dim() == 3, f"Expected vpreds (bs, T, K), got {vpreds.shape}"
    bs, t, k = vpreds.shape
    assert k == num_quantiles, f"num_quantiles mismatch: config {num_quantiles}, pred {k}"
    if taus is None:
        if tau_mode == "iqn":
            taus = torch.rand(bs, t, k, device=vpreds.device, dtype=vpreds.dtype)
        else:
            # fixed midpoints
            taus = (torch.arange(k, device=vpreds.device, dtype=vpreds.dtype) + 0.5) / k  # (K,)
            taus = taus.view(1, 1, k).expand(bs, t, k)
    else:
        assert taus.shape == (bs, t, k), f"taus should have shape (bs, T, K), got {taus.shape}"
    # Expand shapes: pred (bs, T, K, 1), target (bs, T, 1, 1)
    pred = vpreds.unsqueeze(-1)  # (bs, T, K, 1)
    target = returns.unsqueeze(-1).unsqueeze(-1)  # (bs, T, 1, 1)
    diff = target - pred  # (bs, T, K, 1)
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= kappa, 0.5 * diff.pow(2), kappa * (abs_diff - 0.5 * kappa))
    tau = taus.unsqueeze(-1)
    loss = (torch.abs(tau - (diff < 0).float()) * huber).mean(dim=-2)  # mean over target dim
    loss_mask = response_mask.unsqueeze(-1)  # (bs, T, 1)
    loss = verl_F.masked_mean(loss, loss_mask)
    return loss

def compute_categorical_value_loss(
    logits: torch.FloatTensor,
    returns: torch.FloatTensor,
    response_mask: torch.FloatTensor,
    atoms: torch.FloatTensor,
):
    """
    Compute the categorical value loss for C51.
    Projects the scalar returns onto the fixed atoms (support) to create a target distribution,
    then computes the Cross Entropy loss between the predicted logits and the target distribution.
    Args:
        logits: (bs, T, num_atoms) - predicted logits (unnormalized log-probs)
        returns: (bs, T) - scalar target returns
        response_mask: (bs, T) - mask for valid tokens
        atoms: (num_atoms,) - fixed atom values (support)
    """
    # 1. Validation
    bs, t, n_atoms = logits.shape
    assert atoms.shape == (n_atoms,), f"atoms shape mismatch: expected ({n_atoms},), got {atoms.shape}"

    # 2. Project returns onto atoms
    # returns: (bs, T) -> (bs, T, 1)
    target_returns = returns.clamp(min=atoms.min().item(), max=atoms.max().item()).unsqueeze(-1)

    delta_z = atoms[1] - atoms[0]

    # bj: (bs, T, 1) - continuous index of the target return in the atom space
    bj = (target_returns - atoms[0]) / delta_z

    # Calculate probabilities for lower and upper atoms (linear interpolation)
    l = bj.floor().long()
    u = l + 1

    prob_u = bj - l.float()#touyingdaomubiaoqujian
    prob_l = 1.0 - prob_u

    # We need to scatter these probs into target_dist.
    target_probs = torch.zeros(bs, t, n_atoms, device=logits.device, dtype=logits.dtype)

    # Flat view for simpler scatter

    target_probs_flat = target_probs.view(-1, n_atoms) # (N, n_atoms) with N = bs*T
    l_flat = l.view(-1, 1)     # (N, 1)
    u_flat = u.view(-1, 1)     # (N, 1)
    prob_l_flat = prob_l.view(-1, 1) 
    prob_u_flat = prob_u.view(-1, 1)
    prob_l_flat = prob_l_flat.to(target_probs_flat.dtype)
    prob_u_flat = prob_u_flat.to(target_probs_flat.dtype)

    # scatter_add_
    # We clip u_flat to n_atoms-1 to avoid index error? 
    # If u == n_atoms, prob_u SHOULD be 0 because bj <= n_atoms-1 -> bj - floor(bj) ???
    # If return = v_max, bj = n_atoms - 1. l = n_atoms - 1. u = n_atoms.
    # prob_u = (n_atoms - 1) - (n_atoms - 1) = 0.
    # So yes, u's contribution is 0. We can safely clamp u to be valid index for scatter, it won't add anything.
    u_flat = u_flat.clamp(min=0,max=n_atoms-1)#clip
    l_flat = l_flat.clamp(min=0,max=n_atoms-1)

    target_probs_flat.scatter_add_(1, l_flat, prob_l_flat)
    target_probs_flat.scatter_add_(1, u_flat, prob_u_flat)

    target_probs = target_probs_flat.view(bs, t, n_atoms)

    # 3. Compute Loss (KL / CrossEntropy)
    # CE = - sum(p * log q)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss_per_token = -torch.sum(target_probs * log_probs, dim=-1) # (bs, T)

    # 4. Mask and Average
    loss = verl_F.masked_mean(loss_per_token, response_mask)

    return loss

def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
