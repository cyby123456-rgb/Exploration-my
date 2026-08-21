#!/usr/bin/env python3
"""Test whether flip-associated random directions cluster on the unit sphere."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def direction(seed: int, hidden_size: int) -> np.ndarray:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    vector = torch.randn(hidden_size, generator=generator, dtype=torch.float32).numpy()
    return vector / np.linalg.norm(vector)


def mean_pairwise_cosine(vectors: np.ndarray) -> float | None:
    if len(vectors) < 2:
        return None
    gram = vectors @ vectors.T
    return float((gram.sum() - len(vectors)) / (len(vectors) * (len(vectors) - 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()

    summary_path = args.input_jsonl.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hidden_size = int(summary["hidden_size"])
    flip_transition = str(summary["flip_transition"])
    records = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line]
    vectors = {}
    grouped = defaultdict(list)
    for record in records:
        seed = int(record["direction_seed"])
        vectors.setdefault(seed, direction(seed, hidden_size))
        grouped[(str(record["position_fraction"]), str(record["alpha"]))].append(record)

    rng = np.random.default_rng(20260811)
    results = {}
    for key, group in grouped.items():
        by_seed = defaultdict(lambda: [0, 0])  # flips, trials
        for record in group:
            by_seed[int(record["direction_seed"])][1] += 1
            by_seed[int(record["direction_seed"])][0] += int(record["transition"] == flip_transition)
        successful = [seed for seed, (flips, _) in by_seed.items() if flips > 0]
        all_seeds = list(by_seed)
        successful_vectors = np.stack([vectors[seed] for seed in successful]) if successful else np.empty((0, hidden_size))
        observed = mean_pairwise_cosine(successful_vectors)
        permutation_values = []
        if observed is not None:
            for _ in range(args.permutations):
                sampled = rng.choice(all_seeds, size=len(successful), replace=False)
                permutation_values.append(mean_pairwise_cosine(np.stack([vectors[int(seed)] for seed in sampled])))
        results["position=" + key[0] + "/alpha=" + key[1]] = {
            "flip_transition": flip_transition,
            "trials": len(group),
            "successful_directions": len(successful),
            "unique_directions": len(all_seeds),
            "mean_pairwise_cosine_successful": observed,
            "centroid_norm_successful": float(np.linalg.norm(successful_vectors.mean(axis=0))) if len(successful_vectors) else None,
            "permutation_p_value_cosine_greater": (
                float((1 + sum(value >= observed for value in permutation_values)) / (1 + len(permutation_values)))
                if permutation_values
                else None
            ),
        }

    output_path = args.input_jsonl.with_suffix(".direction_geometry.json")
    output_path.write_text(json.dumps({"input": str(args.input_jsonl), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved direction-geometry summary: {output_path}")


if __name__ == "__main__":
    main()
