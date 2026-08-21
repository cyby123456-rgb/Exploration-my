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
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )
        # ray.init(runtime_env={"env_vars": {"RAY_DEBUG": "legacy"}})
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    validation_noise_cfg = None
    trainer_noise_cfg = OmegaConf.select(config, "trainer.validation_noise")
    if trainer_noise_cfg is not None:
        trainer_noise_cfg = OmegaConf.to_container(trainer_noise_cfg, resolve=True)
        try:
            std_value = float(trainer_noise_cfg.get("std", 0.0))
        except (TypeError, ValueError):
            std_value = 0.0
        if std_value > 0:
            validation_noise_cfg = trainer_noise_cfg

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]
    compute_response_ppl = bool(config.data.get("compute_response_ppl", False))
    analysis_enabled = bool(config.analysis.get("enable", False))
    use_rollout_policy_stats = analysis_enabled and validation_noise_cfg is not None
    if compute_response_ppl and (config.rollout.name != "vllm" or config.rollout.mode != "sync"):
        raise ValueError("data.compute_response_ppl requires synchronous vLLM rollout.")
    if use_rollout_policy_stats and (config.rollout.name != "vllm" or config.rollout.mode != "sync"):
        raise ValueError("Noisy generation analysis requires synchronous vllm rollout policy statistics.")
    chosen_logprob_lst = [[] for _ in range(config.data.n_samples)] if analysis_enabled else None
    entropy_lst = [[] for _ in range(config.data.n_samples)] if analysis_enabled else None
    topk_ids_lst = [[] for _ in range(config.data.n_samples)] if analysis_enabled else None
    topk_probs_lst = [[] for _ in range(config.data.n_samples)] if analysis_enabled else None
    topk_logprobs_lst = [[] for _ in range(config.data.n_samples)] if use_rollout_policy_stats else None
    topk_logits_lst = [[] for _ in range(config.data.n_samples)] if analysis_enabled else None
    chosen_logits_lst = [[] for _ in range(config.data.n_samples)] if analysis_enabled else None
    response_ppl_lst = [[] for _ in range(config.data.n_samples)] if compute_response_ppl else None
    response_mean_logprob_lst = [[] for _ in range(config.data.n_samples)] if compute_response_ppl else None
    response_length_lst = [[] for _ in range(config.data.n_samples)] if compute_response_ppl else None

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]

        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)

        data.meta_info = {
            "validate": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": bool(config.rollout.do_sample),
        }
        if validation_noise_cfg is not None:
            noise_cfg = dict(validation_noise_cfg)
            noise_cfg["global_step"] = int(config.trainer.get("global_step", 0))
            data.meta_info["validation_noise"] = noise_cfg
        if use_rollout_policy_stats:
            data.meta_info["return_rollout_policy_stats"] = True
            data.meta_info["analysis_top_k"] = int(config.analysis.top_k)
        elif compute_response_ppl:
            data.meta_info["return_rollout_log_probs"] = True

        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            policy_stats = None
            if analysis_enabled:
                if use_rollout_policy_stats:
                    policy_stats = output
                else:
                    # The worker group can only split a batch evenly across its
                    # data-parallel ranks. Analyze the padded output first, then
                    # remove the synthetic rows to restore alignment with output.
                    output_padded.meta_info["validate"] = data.meta_info["validate"]
                    output_padded.meta_info["recompute_log_prob"] = False
                    output_padded.meta_info["temperature"] = (
                        config.rollout.val_kwargs.temperature if data.meta_info["validate"] else config.rollout.temperature
                    )
                    output_padded.meta_info["analysis_top_k"] = int(config.analysis.top_k)
                    policy_stats_padded = wg.analyze_generation_policy(output_padded)
                    policy_stats = unpad_dataproto(policy_stats_padded, pad_size=pad_size)

            output_texts = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)

                if compute_response_ppl:
                    valid_len = int(valid_response_length.item()) if hasattr(valid_response_length, "item") else int(valid_response_length)
                    token_logprobs = data_item.batch["rollout_log_probs"][:valid_len].float()
                    if valid_len <= 0:
                        mean_logprob = float("nan")
                        ppl = float("nan")
                    else:
                        mean_logprob = float(token_logprobs.mean().item())
                        # The clamp prevents an overflow for exceptionally unlikely responses.
                        ppl = float(np.exp(min(-mean_logprob, 80.0)))
                    response_ppl_lst[n_sample].append(ppl)
                    response_mean_logprob_lst[n_sample].append(mean_logprob)
                    response_length_lst[n_sample].append(valid_len)

                if analysis_enabled:
                    stats_item = policy_stats[i]
                    valid_len = int(valid_response_length.item()) if hasattr(valid_response_length, "item") else int(valid_response_length)
                    if use_rollout_policy_stats:
                        chosen_logprob_lst[n_sample].append(stats_item.batch["rollout_log_probs"][:valid_len].tolist())
                        topk_ids_lst[n_sample].append(stats_item.batch["rollout_topk_ids"][:valid_len].tolist())
                        topk_probs_lst[n_sample].append(stats_item.batch["rollout_topk_probs"][:valid_len].tolist())
                        topk_logprobs_lst[n_sample].append(stats_item.batch["rollout_topk_log_probs"][:valid_len].tolist())
                    else:
                        chosen_logprob_lst[n_sample].append(stats_item.batch["chosen_log_probs"][:valid_len].tolist())
                        entropy_lst[n_sample].append(stats_item.batch["entropys"][:valid_len].tolist())
                        topk_ids_lst[n_sample].append(stats_item.batch["topk_ids"][:valid_len].tolist())
                        topk_probs_lst[n_sample].append(stats_item.batch["topk_probs"][:valid_len].tolist())
                        topk_logits_lst[n_sample].append(stats_item.batch["topk_logits"][:valid_len].tolist())
                        chosen_logits_lst[n_sample].append(stats_item.batch["chosen_logits"][:valid_len].tolist())

            output_lst[n_sample].extend(output_texts)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset["responses"] = output_lst
    if compute_response_ppl:
        dataset["response_ppls"] = np.transpose(np.array(response_ppl_lst, dtype=object), axes=(1, 0)).tolist()
        dataset["response_mean_logprobs"] = np.transpose(
            np.array(response_mean_logprob_lst, dtype=object), axes=(1, 0)
        ).tolist()
        dataset["response_lengths"] = np.transpose(np.array(response_length_lst, dtype=object), axes=(1, 0)).tolist()
    if analysis_enabled:
        dataset["chosen_logprobs"] = np.transpose(np.array(chosen_logprob_lst, dtype=object), axes=(1, 0)).tolist()
        dataset["topk_ids"] = np.transpose(np.array(topk_ids_lst, dtype=object), axes=(1, 0)).tolist()
        dataset["topk_probs"] = np.transpose(np.array(topk_probs_lst, dtype=object), axes=(1, 0)).tolist()
        if use_rollout_policy_stats:
            dataset["topk_logprobs"] = np.transpose(np.array(topk_logprobs_lst, dtype=object), axes=(1, 0)).tolist()
            dataset["analysis_policy_source"] = "noisy_rollout_behavior"
        else:
            dataset["token_entropies"] = np.transpose(np.array(entropy_lst, dtype=object), axes=(1, 0)).tolist()
            dataset["topk_logits"] = np.transpose(np.array(topk_logits_lst, dtype=object), axes=(1, 0)).tolist()
            dataset["chosen_logits"] = np.transpose(np.array(chosen_logits_lst, dtype=object), axes=(1, 0)).tolist()
            dataset["analysis_policy_source"] = "recomputed_actor"

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
