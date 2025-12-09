import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

# from deepscaler.data.utils import load_dataset
from deepscaler.data.dataset_types import TrainDataset, TestDataset


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        answer = example.pop('answer')
        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data
    return process_fn


def read_jsonl(file_name):
    import json
    """
    Read a JSONL file and return a list of dictionaries.
    Args:
        file_name (str): Path to the JSONL file.
    Returns:
        list: List of dictionaries, each representing a line in the JSONL file.
    """
    data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--jsonl_data', default="", type=str,
                       help='List of jsonl files to load and merge')
    args = parser.parse_args()

    local_dir = "parquet/"

    if "stage1" in args.jsonl_data:
        local_dir = local_dir + "stage1/"
    elif "stage2" in args.jsonl_data:
        local_dir = local_dir + "stage2/"
    elif "stage3" in args.jsonl_data:
        local_dir = local_dir + "stage3/"

    # if args.jsonl_data and len(args.jsonl_data) > 0 and os.path.exists(args.jsonl_data[0]):
    #     local_dir = os.path.dirname(args.jsonl_data[0]) 
    jsonl_name = args.jsonl_data.split("/")[-1]
    data_name = jsonl_name.replace(".jsonl", "")
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)
    print(local_dir)
    
    # Initialize datasets    
    # Load and merge all specified JSONL files
    train_datasets = [TrainDataset.DEEPSCALER]
    if os.path.exists(args.jsonl_data):
        train_dataset = read_jsonl(args.jsonl_data)
    else:
        train_dataset = load_dataset(train_datasets[0])

    # test_datasets = [TestDataset.AIME, TestDataset.AMC, TestDataset.MATH, TestDataset.MINERVA, TestDataset.OLYMPIAD_BENCH]
    test_datasets = []
    # test_datasets_data = [load_dataset(d) for d in test_datasets]

    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    # Process and save each test dataset separately
    # for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):
    #     test_data: List[Dict[str, Any]] = []
    #     process_fn = make_map_fn('test')
    #     for idx, example in enumerate(test_data_list):
    #         processed_example = process_fn(example, idx)
    #         if processed_example is not None:
    #             test_data.append(processed_example)
    #
    #     dataset_name = test_dataset.value.lower()
    #     test_df = pd.DataFrame(test_data)
    #     test_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
    #     print(f"{dataset_name} test data size:", len(test_data))

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, f'{data_name}.parquet'))
