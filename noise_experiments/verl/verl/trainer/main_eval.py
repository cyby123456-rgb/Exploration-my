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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict
from math import comb

import multiprocessing as mp
import queue

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.utils.fs import copy_to_local
from verl.trainer.main_ppo import _select_rm_score_fn
from omegaconf import OmegaConf


def _process_item_inner(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]

    if reward_fn is None:
        compute_score_fn = _select_rm_score_fn(data_source, code_continuous=False)
        score_lst = [compute_score_fn(solution_str=r, ground_truth=ground_truth) for r in response_lst]
    else:
        score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]

    n = len(score_lst)
    c = int(np.sum(np.array(score_lst) > 0))
    passk_lst = []
    for k in range(1, n + 1):
        if c == 0:
            passk = 0.0
        elif n - c < k:
            passk = 1.0
        else:
            passk = 1 - comb(n - c, k) / comb(n, k)
        passk_lst.append(passk)

    avg32 = np.mean(score_lst[:32]) if n >= 32 else np.mean(score_lst)
    return data_source, float(np.mean(score_lst)), passk_lst, float(avg32)


def _run_process_item_with_timeout(q, reward_fn, data_source, response_lst, reward_data):
    try:
        result = _process_item_inner(reward_fn, data_source, response_lst, reward_data)
        q.put(("ok", result))
    except Exception as e:
        q.put(("err", repr(e)))


def get_custom_reward_fn(config):
    import importlib
    import importlib.util
    import os
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    # spec = importlib.util.spec_from_file_location("custom_module", file_path)
    # module = importlib.util.module_from_spec(spec)
    # try:
    #     sys.modules["custom_module"] = module
    #     spec.loader.exec_module(module)
    # except Exception as e:
    #     raise RuntimeError(f"Error loading module from '{file_path}'") from e

    module = None
    abs_path = os.path.abspath(file_path)
    repo_dir = os.getcwd()
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    rel_path = os.path.relpath(abs_path, repo_dir)
    if not rel_path.startswith("..") and rel_path.endswith(".py"):
        module_name = rel_path[:-3].replace(os.sep, ".")
        if all(part.isidentifier() for part in module_name.split(".")):
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                raise RuntimeError(
                    f"Error importing reward function module '{module_name}' from '{file_path}'"
                ) from e

    if module is None:
        module_name = f"_verl_custom_reward_{os.path.splitext(os.path.basename(abs_path))[0]}"
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}'") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def _extract_prompt_text(prompt):
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)) and prompt:
        first_message = prompt[0]
        if isinstance(first_message, dict):
            return str(first_message.get("content", ""))
    return str(prompt or "")


def _with_memory_context(extra_info, prompt):
    if isinstance(extra_info, dict):
        prepared = dict(extra_info)
    else:
        prepared = {}
    prepared.setdefault("memory_context", _extract_prompt_text(prompt))
    return prepared


def log_metrics_if_configured(config, metric_dict):
    tracking_cfg = config.get("tracking") or {}
    logger_backends = tracking_cfg.get("logger") or []
    if not logger_backends:
        return None

    from verl.utils.tracking import Tracking

    project_name = tracking_cfg.get("project_name") or "verl-eval"
    experiment_name = tracking_cfg.get("experiment_name") or "main_eval"
    logger = Tracking(
        project_name=project_name,
        experiment_name=experiment_name,
        default_backend=logger_backends,
        config=OmegaConf.to_container(config, resolve=True),
    )
    logger.log(data=metric_dict, step=0)
    return logger


def log_wandb_summary_table(logger, data_source_reward, data_source_avg32, data_source_passk):
    if logger is None or "wandb" not in logger.logger:
        return

    max_k = 0
    for passk_lists in data_source_passk.values():
        for lst in passk_lists:
            if len(lst) > max_k:
                max_k = len(lst)

    columns = ["data_source", "test_score", "test_avg@32", "test_avg@4"]
    columns.extend([f"pass@{k}" for k in range(1, max_k + 1)])

    rows = []
    for data_source in sorted(data_source_reward.keys()):
        rewards = data_source_reward.get(data_source, [])
        avg32_list = data_source_avg32.get(data_source, [])
        avg4_list = data_source_avg4.get(data_source, [])
        passk_lists = data_source_passk.get(data_source, [])

        row = [
            data_source,
            float(np.mean(rewards)) if rewards else None,
            float(np.mean(avg32_list)) if avg32_list else None,
            float(np.mean(avg4_list)) if avg4_list else None,
        ]

        for idx in range(max_k):
            vals = [lst[idx] for lst in passk_lists if len(lst) > idx]
            row.append(float(np.mean(vals)) if vals else None)

        rows.append(row)

    import wandb

    table = wandb.Table(columns=columns, data=rows)
    wandb.log({"eval/summary_table": table}, step=0)


def log_wandb_per_response_ppl(logger, dataset, config):
    """Log PPL generated by main_generation, aligned with each evaluated answer."""
    if logger is None or "wandb" not in logger.logger or "response_ppls" not in dataset.columns:
        return

    include_text = bool((config.get("tracking") or {}).get("log_per_response_ppl_include_text", False))
    columns = [
        "response_index",
        "prompt_index",
        "sample_index",
        "data_source",
        "response_length",
        "mean_logprob",
        "ppl",
    ]
    if include_text:
        columns.extend(["input", "output"])

    rows = []
    response_index = 0
    for prompt_index, row in dataset.iterrows():
        ppls = row["response_ppls"]
        mean_logprobs = row.get("response_mean_logprobs", [None] * len(ppls))
        lengths = row.get("response_lengths", [None] * len(ppls))
        responses = row[config.data.response_key]
        if not (len(ppls) == len(mean_logprobs) == len(lengths) == len(responses)):
            raise ValueError(f"PPL fields are not aligned with responses for prompt row {prompt_index}.")
        for sample_index, (ppl, mean_logprob, length, response) in enumerate(zip(ppls, mean_logprobs, lengths, responses)):
            record = [
                response_index,
                int(prompt_index),
                sample_index,
                row[config.data.data_source_key],
                length,
                mean_logprob,
                ppl,
            ]
            if include_text:
                record.extend([_extract_prompt_text(row.get("prompt", "")), response])
            rows.append(record)
            response_index += 1

    import wandb

    wandb.log({"eval/per_response_ppl": wandb.Table(columns=columns, data=rows)}, step=0)

@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data, extra_info):
    ground_truth = reward_data["ground_truth"]
    rubric_scores = defaultdict(list)
    rubric_na_counts = defaultdict(int)
    rubric_total_counts = defaultdict(int)
    if reward_fn is None:
        compute_score_fn = _select_rm_score_fn(data_source, code_continuous=False)
        if data_source in {"locomo", "longmemeval", "scriptmem_memory_qa"}:
            score_lst = [
                compute_score_fn(solution_str=r, ground_truth=ground_truth, extra_info=extra_info)
                for r in response_lst
            ]
        else:
            score_lst = [compute_score_fn(solution_str=r, ground_truth=ground_truth) for r in response_lst]
        print(f"score_lst: '{score_lst}'")
    else:
        results = [
            reward_fn(
                data_source=data_source,
                solution_str=r,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            for r in response_lst
        ]
        score_lst = []
        for result in results:
            if isinstance(result, dict):
                score_lst.append(float(result["score"]))
                for name, value in result.get("rubric_scores", {}).items():
                    name = str(name)
                    rubric_total_counts[name] += 1
                    if value is None:
                        rubric_na_counts[name] += 1
                    else:
                        rubric_scores[name].append(float(value))
            else:
                score_lst.append(float(result))
    n = len(score_lst)
    c = int(np.sum(np.array(score_lst) > 0))
    passk_lst = []
    for k in range(1, n+1):
        if c == 0:
            passk = 0.0
        elif n - c < k:
            passk = 1.0
        else:
            passk = 1 - comb(n - c, k) / comb(n, k)
        passk_lst.append(passk)
    avg32 = np.mean(score_lst[:32]) if n >= 32 else np.mean(score_lst)
    avg4 = np.mean(score_lst[:4]) if n >= 4 else np.mean(score_lst)
    rubric_means = {name: float(np.mean(values)) for name, values in rubric_scores.items() if values}
    for name, total in rubric_total_counts.items():
        if total:
            rubric_means[f"{name}/na_rate"] = float(rubric_na_counts[name] / total)
    return data_source, np.mean(score_lst), passk_lst, avg32, avg4, rubric_means


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    extra_infos = dataset["extra_info"] if "extra_info" in dataset.columns else [None] * len(dataset)
    prompts = dataset["prompt"] if "prompt" in dataset.columns else [""] * len(dataset)

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    data_source_passk = defaultdict(list)
    data_source_avg32 = defaultdict(list)
    data_source_avg4 = defaultdict(list)
    data_source_rubric = defaultdict(lambda: defaultdict(list))
    compute_score = get_custom_reward_fn(config)

    # Create remote tasks
    remote_tasks = [
        process_item.remote(
            compute_score,
            data_sources[i],
            responses[i],
            reward_model_data[i],
            _with_memory_context(extra_infos[i], prompts[i]),
        )
        for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score, passk_lst, avg32, avg4, rubric_means = ray.get(result_id)
                data_source_reward[data_source].append(score)
                data_source_passk[data_source].append(passk_lst)
                data_source_avg32[data_source].append(avg32)
                data_source_avg4[data_source].append(avg4)
                for name, score_value in rubric_means.items():
                    data_source_rubric[data_source][name].append(score_value)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)
    for data_source, avg32_list in data_source_avg32.items():
        metric_dict[f"test_avg@32/{data_source}"] = float(np.mean(avg32_list))
    for data_source, avg4_list in data_source_avg4.items():
        metric_dict[f"test_avg@4/{data_source}"] = float(np.mean(avg4_list))
    for data_source, passk_lists in data_source_passk.items():
        if not passk_lists:
            continue
        max_len = max(len(lst) for lst in passk_lists)
        for idx in range(max_len):
            vals = [lst[idx] for lst in passk_lists if len(lst) > idx]
            if vals:
                metric_dict[f"test_pass@{idx + 1}/{data_source}"] = float(np.mean(vals))
    for data_source, rubric_values in data_source_rubric.items():
        for rubric_name, values in rubric_values.items():
            metric_dict[f"test_rubric/{data_source}/{rubric_name}"] = float(np.mean(values))
    if "response_ppls" in dataset.columns:
        all_ppls = [float(ppl) for ppls in dataset["response_ppls"] for ppl in ppls if np.isfinite(ppl)]
        all_logprobs = [
            float(logprob)
            for logprobs in dataset.get("response_mean_logprobs", [])
            for logprob in logprobs
            if np.isfinite(logprob)
        ]
        all_lengths = [int(length) for lengths in dataset.get("response_lengths", []) for length in lengths]
        if all_ppls:
            metric_dict["eval/ppl_per_response_mean"] = float(np.mean(all_ppls))
            metric_dict["eval/ppl_per_response_median"] = float(np.median(all_ppls))
        if all_logprobs:
            metric_dict["eval/mean_logprob_per_response"] = float(np.mean(all_logprobs))
        if all_logprobs and len(all_logprobs) == len(all_lengths) and sum(all_lengths) > 0:
            metric_dict["eval/ppl_token_weighted"] = float(np.exp(-np.average(all_logprobs, weights=all_lengths)))

    logger = log_metrics_if_configured(config, metric_dict)
    log_wandb_per_response_ppl(logger, dataset, config)
    print(metric_dict)




if __name__ == "__main__":
    main()
