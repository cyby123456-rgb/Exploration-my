#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-inference script for AIME-24:
• N = --n stochastic roll-outs per prompt, each with a unique, random seed
• 32 roll-outs are evenly divided over 8 GPUs (4 seeds / GPU)
• Each GPU loads the model only once and iterates over its own seed list
• All original global variables & argparse flags are kept unchanged
"""
import os
import json
import re
import random
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse

# --------------------------------------------------------------------------- #
#                               Argument parser                               #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/path/to/model")
parser.add_argument("--t", type=float, default=1.4)
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--n", type=int, default=32)          # roll-outs / prompt
parser.add_argument("--max_length", type=int, default=90000)
parser.add_argument("--experiment_name", type=str, default="Polaris-4B")
parser.add_argument("--output", type=str, default="evaluation/results")
args = parser.parse_args()

# --------------------------------------------------------------------------- #
#                   Original global constants / variables                     #
# --------------------------------------------------------------------------- #
NAME        = args.experiment_name
N           = args.n                          # roll-outs per prompt
AIME_PATH   = "evaluation/benchmarks/aime24.parquet"
assert Path(AIME_PATH).exists(), f"{AIME_PATH} does not exist"
MODEL_PATH  = args.model
MAX_TOKENS  = args.max_length
TEMPERATURE = args.t
TOP_P       = 1.0
TOP_K       = args.k
OUT_PATH    = Path(
    f"{args.output}/{NAME}/aime24-{TEMPERATURE}-{N}-{MAX_TOKENS}-{TOP_K}.jsonl"
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
    """Read parquet file and return a list of prompts (no duplication)."""
    df = pd.read_parquet(filepath)
    samples = [
        {
            "example_id": i,
            "prompt": df.at[i, "prompt"][0]["content"],
            "answer": df.at[i, "reward_model"]["ground_truth"],
        }
        for i in range(len(df))
    ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def extract_boxed_answer(text: str):
    """Extract the last boxed{…} string from a LaTeX-like answer."""
    answers = []
    for piece in text.split("boxed{")[1:]:
        n = 0
        for i, ch in enumerate(piece):
            if ch == "{":
                n += 1
            elif ch == "}":
                n -= 1
                if n < 0:
                    answers.append(piece[: i] if (i + 1 == len(piece) or piece[i + 1] != "%") else piece[: i + 1])
                    break
    return answers[-1] if answers else None



def split_seeds(seeds: list[int], num_workers: int):
    """Round-robin split of the seed list into num_workers chunks."""
    chunks = [[] for _ in range(num_workers)]
    for idx, s in enumerate(seeds):
        chunks[idx % num_workers].append(s)
    return chunks


# --------------------------------------------------------------------------- #
#                           Worker process (one GPU)                          #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
    """
    Each worker runs on a single GPU:

    args_tuple = (samples, seed_list, gpu_id)
    """
    samples, seed_list, gpu_id = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] seeds={seed_list} | loading model...", flush=True)

    llm = LLM(model=MODEL_PATH, enforce_eager=True)
    results = []

    for seed in seed_list:
        sampling = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_TOKENS,
            seed=seed,
        )
        messages = [[{"role": "user", "content": s["prompt"]}] for s in samples]
        outputs = llm.chat(messages, sampling, use_tqdm=True)
        for sample, out in zip(samples, outputs):
            results.append(
                {
                    "example_id": sample["example_id"],
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "seed": seed,
                    "response": out.outputs[0].text,
                }
            )
    return results


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    # 1. Load original prompts
    samples = load_samples(AIME_PATH)

    # 2. Generate N distinct random seeds and split across 8 GPUs
    random_seeds = random.sample(range(2**31 - 1), N)  # unique & shuffled
    num_workers = 8
    seed_chunks = split_seeds(random_seeds, num_workers)

    # 3. Launch workers
    all_results = []
    args_list = [(samples, seed_chunks[gid], gid) for gid in range(num_workers)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(worker_process, tup) for tup in args_list]
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), desc="GPU workers"):
            all_results.extend(fut.result())

    print(f"Total generations collected: {len(all_results)}")  # len(samples) * N

    # 4. Save to disk
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
