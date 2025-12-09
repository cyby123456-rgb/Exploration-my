# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Implement a multiprocess PPOCritic
"""

import itertools
import logging
import os

import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.model import sample_quantile_fractions
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

__all__ = ["DataParallelPPOCritic"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        self.is_distributional = getattr(self.config, "distributional", False)
        self.num_quantiles = getattr(self.config, "num_quantiles", 1)
        self.quantile_huber_kappa = getattr(self.config, "quantile_huber_kappa", 1.0)
        self.quantile_mode = getattr(self.config, "quantile_mode", "iqn")
        if self.is_distributional:
            if self.quantile_mode == "fixed" and not hasattr(self.critic_module, "qr_head"):
                raise AttributeError("Quantile mode fixed requires critic_module.qr_head for QR-DQN.")
            if self.quantile_mode == "c51" and not hasattr(self.critic_module, "c51_head"):
                raise AttributeError("Quantile mode c51 requires critic_module.c51_head for categorical C51.")
            if self.quantile_mode not in ["fixed", "c51"] and not hasattr(self.critic_module, "iqn_head"):
                raise AttributeError("IQN quantile mode requires critic_module.iqn_head for IQN.")
        self.c51_v_min = getattr(self.config, "c51_v_min", -10.0)
        self.c51_v_max = getattr(self.config, "c51_v_max", 10.0)
        print(f"Critic use_remove_padding={self.use_remove_padding}, distributional={self.is_distributional}")
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        values = None
        hidden_states = None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=self.is_distributional,
                    return_dict=True,
                )  # prevent model thinks we are generating
                
                logits_rmpad = output.logits
                hidden_rmpad = output.hidden_states[-1] if self.is_distributional else None
                if not self.is_distributional:
                    values_rmpad = logits_rmpad.squeeze(0)  # (total_nnz)
                else:
                    # (1, total_nnz, K) -> (total_nnz, K)
                    hidden_rmpad = hidden_rmpad.squeeze(0)  # (total_nnz, hidden)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    if not self.is_distributional:
                        values_rmpad = gather_outpus_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if self.is_distributional:
                        hidden_rmpad = gather_outpus_and_unpad(hidden_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                #values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                if not self.is_distributional:
                    values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                    values = values[:, -response_length - 1 : -1]
                else:
                    hidden_states = pad_input(hidden_rmpad, indices=indices, batch=batch, seqlen=seqlen)
                    hidden_states = hidden_states[:, -response_length - 1 : -1, :]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=self.is_distributional,
                    return_dict=True,
                )  # prevent model thinks we are generating
                #values = values[:, -response_length - 1 : -1].squeeze(-1)
                
                if not self.is_distributional:
                    values = output.logits
                    values = values[:, -response_length - 1 : -1]
                else:
                    hidden_states = output.hidden_states[-1][:, -response_length - 1 : -1, :]
            if not self.is_distributional:
                if values is None:
                    raise RuntimeError("Non-distributional critic did not produce values.")
                values = values.squeeze(-1)
                return values

            # distributional IQN path
            if hidden_states is None:
                raise RuntimeError("Distributional critic did not produce hidden_states.")
            bsz, seq_len, _ = hidden_states.shape
            if self.quantile_mode == "fixed" and hasattr(self.critic_module, "qr_head"):
                quantiles = self.critic_module.qr_head(hidden_states)
                return quantiles, None
            if self.quantile_mode == "c51" and hasattr(self.critic_module, "c51_head"):
                c51_logits = self.critic_module.c51_head(hidden_states)
                return c51_logits, None
            taus = sample_quantile_fractions(
                batch=bsz,
                seq_len=seq_len,
                num_quantiles=self.num_quantiles,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                mode=self.quantile_mode,
            )
            quantiles = self.critic_module.iqn_head(hidden_states, taus)
            return quantiles, taus

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch) # return (quan, taus)
            values_lst.append(values)
        if self.is_distributional:
            # tolerate any extra items while only keeping quantiles
            quantiles = []
            for item in values_lst:
                if isinstance(item, (tuple, list)):
                    quantiles.append(item[0]) # item(quan, taus) quan
                else:
                    quantiles.append(item)
            values = torch.concat(quantiles, dim=0)
        else:
            values = torch.concat(values_lst, dim=0)
        #values = torch.concat(values_lst, dim=0)
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        #values = values * attention_mask[:, -response_length - 1 : -1]
        response_mask = attention_mask[:, -response_length - 1 : -1]
        if self.is_distributional:
            if self.quantile_mode == "c51":
                atoms = torch.linspace(self.c51_v_min, self.c51_v_max, values.size(-1), device=values.device, dtype=values.dtype)
                probs = torch.softmax(values, dim=-1)
                expect = (probs * atoms.view(1, 1, -1)).sum(dim=-1)
                values = expect * response_mask
                values_mean = values
            else:
                values = values * response_mask.unsqueeze(-1)
                values_mean = values.mean(dim=-1)
        else:
            values = values * response_mask
            values_mean = values

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]
            values_mean = values_mean[revert_indices]

        return values_mean

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

                self.critic_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all devices
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # critic device is cpu when using offload
                    responses = data["responses"]
                    attention_mask = data["attention_mask"]
                    values = data["values"]
                    returns = data["returns"]
                    response_length = responses.size(1)

                    response_mask = attention_mask[:, -response_length - 1 : -1]

                    vpreds = self._forward_micro_batch(data)

                    # assert not torch.any(torch.isnan(vpreds)).item()

                    #vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                    #    vpreds=vpreds,
                    #    values=values,
                    #    returns=returns,
                    #    response_mask=response_mask,
                    #    cliprange_value=self.config.cliprange_value,
                    #)
                    if self.is_distributional:
                        vpreds, taus = vpreds
                        quantile_mask = response_mask.unsqueeze(-1).bool()
                        flat_quantiles = torch.masked_select(vpreds.detach(), quantile_mask)
                        if flat_quantiles.numel() > 0:
                            append_to_dict(
                                metrics,
                                {
                                    "critic/quantile_min": flat_quantiles.min().item(),
                                    "critic/quantile_max": flat_quantiles.max().item(),
                                    "critic/quantile_var": flat_quantiles.var(unbiased=False).item(),
                                    "critic/quantile_std": flat_quantiles.std(unbiased=False).item(),
                                },
                            )
                        vf_loss = core_algos.compute_quantile_value_loss(
                            vpreds=vpreds,  # (bs, T, K)
                            returns=returns,  # scalar targets
                            response_mask=response_mask,
                            num_quantiles=self.num_quantiles,
                            tau_mode=self.quantile_mode,  # iqn or fixed
                            kappa=self.quantile_huber_kappa,
                            taus=taus,
                        )
                        vf_clipfrac = torch.tensor(0.0, device=vpreds.device)  # not used for dist. loss
                        vpred_mean = masked_mean(vpreds.mean(dim=-1), response_mask).detach().item()  # mean of quantiles
                    else:
                        vpreds = vpreds.squeeze(-1)  # (bs, T)
                        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                            vpreds=vpreds,
                            values=values,
                            returns=returns,
                            response_mask=response_mask,
                            cliprange_value=self.config.cliprange_value,
                        )
                        vpred_mean = masked_mean(vpreds, response_mask).detach().item()  # mean value

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = vf_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = vf_loss / self.gradient_accumulation

                    loss.backward()

                    data = {
                        "critic/vf_loss": vf_loss.detach().item(),
                        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                        #"critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                        "critic/vpred_mean": vpred_mean,
                    }

                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.critic_optimizer.zero_grad()
        return metrics
