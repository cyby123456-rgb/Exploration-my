#!/usr/bin/env python3
"""Fixed-state Monte-Carlo probes for hidden-state intervention directions.

The input is a clean generation parquet containing ``prompt``, ``responses``,
``data_source`` and ``reward_model``. For an eligible row the script chooses
one response with a requested baseline outcome (wrong or correct), replays a
prefix of that response, and injects
exactly one unit direction at the final prefix token of one decoder layer.
Each intervention is then greedily continued and scored.

This is intentionally a Hugging Face, batch-size-one tool.  The existing vLLM
validation-noise path adds fresh noise throughout a rollout and therefore is
not suitable for a fixed-state causal probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
VERL_ROOT = Path(os.environ.get("VERL_ROOT", ROOT / "verl")).resolve()
sys.path.insert(0, str(VERL_ROOT))

from verl.models.transformers.noise_injection import _get_decoder_layers, register_one_shot_hidden_perturbation
from verl.trainer.main_ppo import _select_rm_score_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--model", required=True, help="HF checkpoint path or identifier")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alphas", default="0.025,0.05", help="Comma-separated relative radii")
    parser.add_argument("--directions", type=int, default=64)
    parser.add_argument(
        "--direction-batch-size",
        type=int,
        default=8,
        help="Number of independent directions to continue concurrently (reduce if CUDA OOMs)",
    )
    parser.add_argument("--max-states", type=int, default=48)
    parser.add_argument("--min-clean-correct", type=int, default=8)
    parser.add_argument("--max-clean-correct", type=int, default=23)
    parser.add_argument(
        "--base-outcome",
        choices=("wrong", "correct"),
        default="wrong",
        help="Probe W->R directions from a wrong state, or R->W directions from a correct state.",
    )
    parser.add_argument("--prefix-fraction", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--log-every", type=int, default=10, help="Print and flush progress every N interventions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_alphas(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("--alphas must contain one or more non-negative values")
    return alphas


def as_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    return value


def score_response(record: dict[str, Any], response: str) -> float:
    score_fn = _select_rm_score_fn(str(record["data_source"]))
    result = score_fn(solution_str=response, ground_truth=record["reward_model"]["ground_truth"])
    if isinstance(result, dict):
        result = result.get("score", 0.0)
    return float(result)


def prompt_ids(tokenizer, prompt: Any, max_input_tokens: int) -> list[int]:
    prompt = as_python(prompt)
    if isinstance(prompt, list):
        ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    elif isinstance(prompt, str):
        ids = tokenizer(prompt, add_special_tokens=True).input_ids
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")
    return list(ids[-max_input_tokens:])


def response_ids(tokenizer, response: str) -> list[int]:
    # Existing generation parquets store decoded text rather than token IDs.
    # New collectors should preserve response_token_ids to avoid this fallback.
    return list(tokenizer(str(response), add_special_tokens=False).input_ids)


@torch.inference_mode()
def greedy_continue_batch(
    model,
    tokenizer,
    input_ids: list[int],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    inputs = torch.tensor([input_ids] * batch_size, device=device, dtype=torch.long)
    attention_mask = torch.ones_like(inputs)
    generated = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return [tokenizer.decode(row[inputs.shape[1] :], skip_special_tokens=True) for row in generated]


def greedy_continue(model, tokenizer, input_ids: list[int], max_new_tokens: int, device: torch.device) -> str:
    return greedy_continue_batch(model, tokenizer, input_ids, 1, max_new_tokens, device)[0]


def select_states(df: pd.DataFrame, args: argparse.Namespace, tokenizer) -> list[dict[str, Any]]:
    selected = []
    for row_index, raw_record in enumerate(df.to_dict("records")):
        record = {key: as_python(value) for key, value in raw_record.items()}
        responses = record.get("responses") or []
        scores = [score_response(record, str(response)) for response in responses]
        correct_count = sum(score > 0 for score in scores)
        if not args.min_clean_correct <= correct_count <= args.max_clean_correct:
            continue
        candidate_indices = [
            idx for idx, score in enumerate(scores) if (score > 0 if args.base_outcome == "correct" else score <= 0)
        ]
        if not candidate_indices:
            continue
        response_index = candidate_indices[0]
        p_ids = prompt_ids(tokenizer, record["prompt"], args.max_input_tokens)
        r_ids = response_ids(tokenizer, str(responses[response_index]))
        prefix_response_length = max(1, int(len(r_ids) * args.prefix_fraction))
        prefix_response_ids = r_ids[:prefix_response_length]
        if not prefix_response_ids:
            continue
        prefix_ids = (p_ids + prefix_response_ids)[-args.max_input_tokens:]
        selected.append(
            {
                "record": record,
                "row_index": row_index,
                "response_index": response_index,
                "clean_correct_count": correct_count,
                "clean_response": str(responses[response_index]),
                "prefix_ids": prefix_ids,
                "prefix_response_tokens": prefix_response_length,
            }
        )
        if len(selected) >= args.max_states:
            break
    return selected


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    if not 0 < args.prefix_fraction <= 1:
        raise ValueError("--prefix-fraction must be in (0, 1]")
    if args.directions <= 0 or args.max_states <= 0 or args.direction_batch_size <= 0:
        raise ValueError("--directions, --max-states, and --direction-batch-size must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    started_at = time.monotonic()
    print(f"[direction-probe] loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[direction-probe] loading model on {device} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()
    hidden_size = int(model.config.hidden_size)
    layer_count = len(_get_decoder_layers(model))
    if not 0 <= args.layer < layer_count:
        raise ValueError(f"--layer must be in [0, {layer_count - 1}], got {args.layer}.")

    print(f"[direction-probe] reading input: {args.input_parquet}", flush=True)
    df = pd.read_parquet(args.input_parquet)
    required_columns = {"prompt", "responses", "data_source", "reward_model"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {sorted(missing)}")
    states = select_states(df, args, tokenizer)
    if not states:
        raise RuntimeError(f"No eligible {args.base_outcome} trajectories found; broaden the clean-correctness range.")

    planned_interventions = len(states) * len(alphas) * args.directions
    print(
        f"[direction-probe] selected {len(states)} candidate states; "
        f"planned maximum={planned_interventions} interventions "
        f"({len(alphas)} alpha x {args.directions} directions/state, batch={args.direction_batch_size})",
        flush=True,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    counts: dict[float, Counter] = defaultdict(Counter)
    state_counts: dict[int, dict[float, Counter]] = defaultdict(lambda: defaultdict(Counter))
    expected_correct = args.base_outcome == "correct"
    base_label = "right" if expected_correct else "wrong"
    other_label = "wrong" if expected_correct else "right"
    flip_transition = f"{base_label}_to_{other_label}"
    stay_transition = f"{base_label}_to_{base_label}"
    accepted_states = 0
    completed_interventions = 0

    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for state_index, state in enumerate(states):
            record = state["record"]
            print(
                f"[direction-probe] state {state_index + 1}/{len(states)}: "
                f"checking unperturbed continuation",
                flush=True,
            )
            baseline_response = greedy_continue(model, tokenizer, state["prefix_ids"], args.max_new_tokens, device)
            baseline_score = score_response(record, baseline_response)
            # Exclude trajectories whose greedy replay does not preserve the
            # requested fixed base state after prefix truncation.
            if (baseline_score > 0) != expected_correct:
                print(
                    f"[direction-probe] state {state_index + 1}: skipped "
                    f"(baseline is not {args.base_outcome})",
                    flush=True,
                )
                continue
            accepted_states += 1
            for alpha in alphas:
                for chunk_start in range(0, args.directions, args.direction_batch_size):
                    direction_indices = list(range(chunk_start, min(chunk_start + args.direction_batch_size, args.directions)))
                    direction_seeds = [
                        int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item()) for _ in direction_indices
                    ]
                    directions = []
                    direction_hashes = []
                    for direction_seed in direction_seeds:
                        direction_generator = torch.Generator(device="cpu").manual_seed(direction_seed)
                        direction = torch.randn(hidden_size, generator=direction_generator, dtype=torch.float32)
                        directions.append(direction)
                        direction_hashes.append(hashlib.sha256(direction.numpy().tobytes()).hexdigest())
                    direction_batch = torch.stack(directions)
                    handle, diagnostics = register_one_shot_hidden_perturbation(
                        model,
                        layer_idx=args.layer,
                        direction=direction_batch,
                        target_position=-1,
                        alpha=alpha,
                    )
                    try:
                        perturbed_responses = greedy_continue_batch(
                            model,
                            tokenizer,
                            state["prefix_ids"],
                            len(direction_indices),
                            args.max_new_tokens,
                            device,
                        )
                    finally:
                        handle.remove()
                    if not diagnostics["applied"]:
                        raise RuntimeError("The one-shot hook did not fire during generation.")
                    for direction_index, direction_seed, direction_hash, perturbed_response in zip(
                        direction_indices, direction_seeds, direction_hashes, perturbed_responses, strict=True
                    ):
                        perturbed_score = score_response(record, perturbed_response)
                        transition = flip_transition if (perturbed_score > 0) != expected_correct else stay_transition
                        counts[alpha][transition] += 1
                        state_counts[state_index][alpha][transition] += 1
                        output.write(
                            json.dumps(
                                {
                                "state_index": state_index,
                                "row_index": state["row_index"],
                                "response_index": state["response_index"],
                                "data_source": record["data_source"],
                                "ground_truth": record["reward_model"]["ground_truth"],
                                "clean_correct_count": state["clean_correct_count"],
                                "base_outcome": args.base_outcome,
                                "prefix_response_tokens": state["prefix_response_tokens"],
                                "prefix_token_count": len(state["prefix_ids"]),
                                "layer": diagnostics["layer_idx"],
                                "alpha": alpha,
                                "direction_index": direction_index,
                                "direction_seed": direction_seed,
                                "direction_sha256": direction_hash,
                                "prefix_token_ids": state["prefix_ids"],
                                "baseline_score": baseline_score,
                                "perturbed_score": perturbed_score,
                                "transition": transition,
                                "baseline_response": baseline_response,
                                "perturbed_response": perturbed_response,
                                "diagnostics": diagnostics,
                                },
                                ensure_ascii=False,
                                default=jsonable,
                            )
                            + "\n"
                        )
                        completed_interventions += 1
                        if completed_interventions % args.log_every == 0:
                            output.flush()
                            elapsed_seconds = time.monotonic() - started_at
                            rate = completed_interventions / elapsed_seconds if elapsed_seconds else 0.0
                            print(
                                f"[direction-probe] {completed_interventions}/{planned_interventions} interventions; "
                                f"state={state_index + 1}/{len(states)}, alpha={alpha}, "
                                f"{flip_transition}={counts[alpha][flip_transition]}/{sum(counts[alpha].values())}; "
                                f"{rate:.3f} interventions/s",
                                flush=True,
                            )
        output.flush()

    summary = {
        "input_parquet": str(args.input_parquet),
        "model": args.model,
        "layer": args.layer,
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "candidate_states": len(states),
        "base_outcome": args.base_outcome,
        "accepted_base_states": accepted_states,
        "directions_per_state": args.directions,
        "alphas": alphas,
        "results": {
            str(alpha): {
                **dict(counter),
                "p_flip": counter[flip_transition] / sum(counter.values()) if counter else None,
            }
            for alpha, counter in counts.items()
        },
        "per_state_results": {
            str(state_index): {
                str(alpha): {
                    **dict(counter),
                    "p_flip": counter[flip_transition] / sum(counter.values()) if counter else None,
                }
                for alpha, counter in per_alpha.items()
            }
            for state_index, per_alpha in state_counts.items()
        },
        "notes": [
            "Primary continuation is greedy to isolate intervention directions from decoding randomness.",
            "For base_outcome=wrong, p_flip is W->R; for base_outcome=correct, p_flip is R->W.",
            "Existing generation parquets only store response text; prefixes are retokenized. New trajectory collectors should store response token IDs.",
            "This script injects once at the final token of a teacher-forced prefix, not throughout rollout.",
        ],
    }
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[direction-probe] complete: {completed_interventions} interventions, "
        f"elapsed={time.monotonic() - started_at:.1f}s, summary={summary_path}",
        flush=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
