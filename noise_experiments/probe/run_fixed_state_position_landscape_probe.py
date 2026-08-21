#!/usr/bin/env python3
"""Probe one layer at several positions of the same clean rollout.

Unlike ``run_fixed_state_direction_probe.py``, this program builds one global
bank of random directions and reuses it at every selected trajectory position
and every alpha.  This makes the effects of a particular direction comparable
across positions and perturbation radii.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_fixed_state_direction_probe import (
    _get_decoder_layers,
    as_python,
    greedy_continue,
    greedy_continue_batch,
    jsonable,
    prompt_ids,
    register_one_shot_hidden_perturbation,
    response_ids,
    score_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--position-fractions", default="0.25,0.5,0.75")
    parser.add_argument("--alphas", default="0.025,0.05")
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--direction-batch-size", type=int, default=8)
    parser.add_argument("--max-states", type=int, default=12)
    parser.add_argument("--min-clean-correct", type=int, default=8)
    parser.add_argument("--max-clean-correct", type=int, default=23)
    parser.add_argument("--base-outcome", choices=("wrong", "correct"), default="wrong")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_float_list(value: str, name: str, *, lower: float, upper: float | None = None) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < lower or (upper is not None and item > upper) for item in values):
        interval = f"[{lower}, {upper}]" if upper is not None else f"[{lower}, inf)"
        raise ValueError(f"{name} must contain values in {interval}")
    return list(dict.fromkeys(values))


def select_trajectories(df: pd.DataFrame, args: argparse.Namespace, tokenizer) -> tuple[list[dict], Counter, Counter]:
    selected = []
    correct_count_histogram: Counter = Counter()
    exclusion_counts: Counter = Counter()
    for row_index, raw_record in enumerate(df.to_dict("records")):
        record = {key: as_python(value) for key, value in raw_record.items()}
        responses = record.get("responses") or []
        scores = [score_response(record, str(response)) for response in responses]
        clean_correct_count = sum(score > 0 for score in scores)
        correct_count_histogram[clean_correct_count] += 1
        if not args.min_clean_correct <= clean_correct_count <= args.max_clean_correct:
            exclusion_counts["outside_correct_count_range"] += 1
            continue
        candidates = [
            index
            for index, score in enumerate(scores)
            if (score > 0 if args.base_outcome == "correct" else score <= 0)
        ]
        if not candidates:
            exclusion_counts["no_requested_base_outcome"] += 1
            continue
        response_index = candidates[0]
        token_ids = response_ids(tokenizer, str(responses[response_index]))
        if not token_ids:
            exclusion_counts["empty_selected_response"] += 1
            continue
        selected.append(
            {
                "record": record,
                "row_index": row_index,
                "response_index": response_index,
                "clean_correct_count": clean_correct_count,
                "prompt_ids": prompt_ids(tokenizer, record["prompt"], args.max_input_tokens),
                "response_token_ids": token_ids,
            }
        )
        if len(selected) >= args.max_states:
            break
    return selected, correct_count_histogram, exclusion_counts


def build_direction_bank(hidden_size: int, count: int, seed: int) -> list[dict]:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    bank = []
    for direction_id in range(count):
        direction_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        generator = torch.Generator(device="cpu").manual_seed(direction_seed)
        vector = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
        bank.append(
            {
                "direction_id": direction_id,
                "direction_seed": direction_seed,
                "direction_sha256": hashlib.sha256(vector.numpy().tobytes()).hexdigest(),
                "vector": vector,
            }
        )
    return bank


def main() -> None:
    args = parse_args()
    fractions = parse_float_list(args.position_fractions, "--position-fractions", lower=0.0, upper=1.0)
    alphas = parse_float_list(args.alphas, "--alphas", lower=0.0)
    if args.directions <= 0 or args.direction_batch_size <= 0 or args.max_states <= 0 or args.log_every <= 0:
        raise ValueError("--directions, --direction-batch-size, --max-states, and --log-every must be positive")

    started_at = time.monotonic()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
    ).to(device).eval()
    hidden_size = int(model.config.hidden_size)
    layer_count = len(_get_decoder_layers(model))
    if not 0 <= args.layer < layer_count:
        raise ValueError(f"--layer must be in [0, {layer_count - 1}], got {args.layer}")

    df = pd.read_parquet(args.input_parquet)
    required_columns = {"prompt", "responses", "data_source", "reward_model"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing)}")
    trajectories, correct_count_histogram, exclusion_counts = select_trajectories(df, args, tokenizer)
    if not trajectories:
        histogram = ", ".join(f"{count_correct}:{count}" for count_correct, count in sorted(correct_count_histogram.items()))
        raise RuntimeError(
            f"No eligible {args.base_outcome} trajectories found. "
            f"clean_correct_count histogram (correct_responses:questions) = {{{histogram}}}; "
            f"selection range=[{args.min_clean_correct}, {args.max_clean_correct}], "
            f"exclusions={dict(exclusion_counts)}."
        )
    direction_bank = build_direction_bank(hidden_size, args.directions, args.seed)

    expected_correct = args.base_outcome == "correct"
    base_label, other_label = ("right", "wrong") if expected_correct else ("wrong", "right")
    flip_transition = f"{base_label}_to_{other_label}"
    stay_transition = f"{base_label}_to_{base_label}"
    planned = len(trajectories) * len(fractions) * len(alphas) * args.directions
    print(
        f"[position-landscape] trajectories={len(trajectories)}, positions={fractions}, "
        f"alphas={alphas}, shared directions={args.directions}, planned_max={planned}",
        flush=True,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    accepted_positions = 0
    completed = 0
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for trajectory_id, trajectory in enumerate(trajectories):
            record = trajectory["record"]
            response_ids_for_trajectory = trajectory["response_token_ids"]
            for position_index, fraction in enumerate(fractions):
                prefix_response_tokens = max(1, min(len(response_ids_for_trajectory), round(len(response_ids_for_trajectory) * fraction)))
                prefix_ids = (trajectory["prompt_ids"] + response_ids_for_trajectory[:prefix_response_tokens])[-args.max_input_tokens :]
                baseline_response = greedy_continue(model, tokenizer, prefix_ids, args.max_new_tokens, device)
                baseline_score = score_response(record, baseline_response)
                if (baseline_score > 0) != expected_correct:
                    print(
                        f"[position-landscape] trajectory={trajectory_id}, fraction={fraction}: baseline skipped",
                        flush=True,
                    )
                    continue
                accepted_positions += 1
                for alpha in alphas:
                    alpha_key = str(alpha)
                    for start in range(0, len(direction_bank), args.direction_batch_size):
                        chunk = direction_bank[start : start + args.direction_batch_size]
                        direction_batch = torch.stack([item["vector"] for item in chunk])
                        handle, diagnostics = register_one_shot_hidden_perturbation(
                            model=model,
                            layer_idx=args.layer,
                            direction=direction_batch,
                            target_position=-1,
                            alpha=alpha,
                        )
                        try:
                            perturbed_responses = greedy_continue_batch(
                                model, tokenizer, prefix_ids, len(chunk), args.max_new_tokens, device
                            )
                        finally:
                            handle.remove()
                        if not diagnostics["applied"]:
                            raise RuntimeError("The one-shot hook did not fire during generation")
                        for direction, perturbed_response in zip(chunk, perturbed_responses, strict=True):
                            perturbed_score = score_response(record, perturbed_response)
                            transition = flip_transition if (perturbed_score > 0) != expected_correct else stay_transition
                            counts[str(fraction)][alpha_key][transition] += 1
                            output.write(
                                json.dumps(
                                    {
                                        "trajectory_id": trajectory_id,
                                        "row_index": trajectory["row_index"],
                                        "response_index": trajectory["response_index"],
                                        "data_source": record["data_source"],
                                        "base_outcome": args.base_outcome,
                                        "clean_correct_count": trajectory["clean_correct_count"],
                                        "position_index": position_index,
                                        "position_fraction": fraction,
                                        "prefix_response_tokens": prefix_response_tokens,
                                        "prefix_token_count": len(prefix_ids),
                                        "layer": diagnostics["layer_idx"],
                                        "alpha": alpha,
                                        "direction_id": direction["direction_id"],
                                        "direction_seed": direction["direction_seed"],
                                        "direction_sha256": direction["direction_sha256"],
                                        "baseline_score": baseline_score,
                                        "perturbed_score": perturbed_score,
                                        "transition": transition,
                                        "diagnostics": diagnostics,
                                    },
                                    ensure_ascii=False,
                                    default=jsonable,
                                )
                                + "\n"
                            )
                            completed += 1
                            if completed % args.log_every == 0:
                                output.flush()
                                print(f"[position-landscape] {completed}/{planned} interventions", flush=True)

    summary = {
        "input_parquet": str(args.input_parquet),
        "model": args.model,
        "layer": args.layer,
        "hidden_size": hidden_size,
        "base_outcome": args.base_outcome,
        "flip_transition": flip_transition,
        "trajectories": len(trajectories),
        "accepted_positions": accepted_positions,
        "position_fractions": fractions,
        "alphas": alphas,
        "directions": args.directions,
        "direction_bank_seed": args.seed,
        "clean_correct_count_histogram": dict(correct_count_histogram),
        "selection_exclusions": dict(exclusion_counts),
        "results": {fraction: {alpha: dict(counter) for alpha, counter in by_alpha.items()} for fraction, by_alpha in counts.items()},
        "notes": [
            "The same global direction bank is reused across every trajectory, position, and alpha.",
            "Use analyze_direction_landscape.py to test whether flip-associated directions cluster on the unit sphere.",
            "Positions are based on retokenized response text; they are approximate if the source parquet lacks response token IDs.",
        ],
    }
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[position-landscape] completed={completed}; summary={summary_path}; elapsed={time.monotonic() - started_at:.1f}s")


if __name__ == "__main__":
    main()
