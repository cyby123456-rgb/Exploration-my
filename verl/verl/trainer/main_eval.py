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

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.utils.fs import copy_to_local
from verl.trainer.main_ppo import _select_rm_score_fn
from omegaconf import OmegaConf

def get_custom_reward_fn(config):
    import importlib.util
    import os
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
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

    columns = ["data_source", "test_score", "test_avg@32"]
    columns.extend([f"pass@{k}" for k in range(1, max_k + 1)])

    rows = []
    for data_source in sorted(data_source_reward.keys()):
        rewards = data_source_reward.get(data_source, [])
        avg32_list = data_source_avg32.get(data_source, [])
        passk_lists = data_source_passk.get(data_source, [])

        row = [
            data_source,
            float(np.mean(rewards)) if rewards else None,
            float(np.mean(avg32_list)) if avg32_list else None,
        ]

        for idx in range(max_k):
            vals = [lst[idx] for lst in passk_lists if len(lst) > idx]
            row.append(float(np.mean(vals)) if vals else None)

        rows.append(row)

    import wandb

    table = wandb.Table(columns=columns, data=rows)
    wandb.log({"eval/summary_table": table}, step=0)

@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    if reward_fn is None:
        compute_score_fn = _select_rm_score_fn(data_source)
        score_lst = [compute_score_fn(solution_str=r, ground_truth=ground_truth) for r in response_lst]
        print(f"score_lst: '{score_lst}'")
    else:
        score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
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
    return data_source, np.mean(score_lst), passk_lst, avg32


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    data_source_passk = defaultdict(list)
    data_source_avg32 = defaultdict(list)
    compute_score = get_custom_reward_fn(config)

    # Create remote tasks
    remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score, passk_lst, avg32 = ray.get(result_id)
                data_source_reward[data_source].append(score)
                data_source_passk[data_source].append(passk_lst)
                data_source_avg32[data_source].append(avg32)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)
    for data_source, avg32_list in data_source_avg32.items():
        metric_dict[f"test_avg@32/{data_source}"] = float(np.mean(avg32_list))
    for data_source, passk_lists in data_source_passk.items():
        if not passk_lists:
            continue
        max_len = max(len(lst) for lst in passk_lists)
        for idx in range(max_len):
            vals = [lst[idx] for lst in passk_lists if len(lst) > idx]
            if vals:
                metric_dict[f"test_pass@{idx + 1}/{data_source}"] = float(np.mean(vals))
    log_metrics_if_configured(config, metric_dict)
    print(metric_dict)




if __name__ == "__main__":
    main()
