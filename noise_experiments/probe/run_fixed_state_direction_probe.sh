#!/usr/bin/env bash
# Fixed-state causal direction probe.  This is self-contained for rjob:
# edit the experiment block below, then launch this file directly.
set -euo pipefail
set -x

# ================= Environment =================
REPO_ROOT="/mnt/shared-storage-user/liujinyi/test123/Noise-Experiment"
VERL_ROOT="${REPO_ROOT}/verl"
CONDA_BASE="/root/miniconda3"
CONDA_ENV="test123"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# The probe is a single-process Hugging Face generation job, so it uses one GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

# ================= Experiment (edit here) =================
EXPERIMENT_NAME="r1-1p5b-32-8k-grpo-clean-seed1002"
DATASET_TAG="aime24"

# This must be a completed *clean generation* parquet. It must contain
# prompt, responses, data_source, and reward_model columns.
CLEAN_PARQUET="${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}/${DATASET_TAG}-1.0-32-8192--1.parquet"
MODEL_PATH="${REPO_ROOT}/model_merged/grpo-8k"

# Probe settings. Run base_outcome=wrong for W->R, then base_outcome=correct
# in a separate rjob for R->W; p_flip has a different meaning in the two runs.
LAYER=24
BASE_OUTCOME="wrong"
ALPHAS="0.025,0.05"
DIRECTIONS=64
DIRECTION_BATCH_SIZE=8
MAX_STATES=48
MIN_CLEAN_CORRECT=8
MAX_CLEAN_CORRECT=23
PREFIX_FRACTION=0.5
MAX_NEW_TOKENS=2048
MAX_INPUT_TOKENS=8192
PROBE_SEED=20260810
DTYPE="bfloat16"

OUTPUT_DIR="${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}/fixed_state_probe"
OUTPUT_JSONL="${OUTPUT_DIR}/${DATASET_TAG}-layer${LAYER}-${BASE_OUTCOME}-seed${PROBE_SEED}.jsonl"

mkdir -p "${OUTPUT_DIR}"

echo "======================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Clean parquet: ${CLEAN_PARQUET}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_JSONL}"
echo "Layer: ${LAYER}; base outcome: ${BASE_OUTCOME}"
echo "Directions: ${DIRECTIONS}; alphas: ${ALPHAS}; probe seed: ${PROBE_SEED}"
echo "======================================"

if [[ ! -f "${CLEAN_PARQUET}" ]]; then
    echo "Clean generation parquet not found: ${CLEAN_PARQUET}" >&2
    exit 1
fi

python3 "${REPO_ROOT}/noise_experiments/probe/run_fixed_state_direction_probe.py" \
    --input-parquet "${CLEAN_PARQUET}" \
    --model "${MODEL_PATH}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --layer "${LAYER}" \
    --base-outcome "${BASE_OUTCOME}" \
    --alphas "${ALPHAS}" \
    --directions "${DIRECTIONS}" \
    --direction-batch-size "${DIRECTION_BATCH_SIZE}" \
    --max-states "${MAX_STATES}" \
    --min-clean-correct "${MIN_CLEAN_CORRECT}" \
    --max-clean-correct "${MAX_CLEAN_CORRECT}" \
    --prefix-fraction "${PREFIX_FRACTION}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --seed "${PROBE_SEED}" \
    --dtype "${DTYPE}" \
    "$@"

echo "Fixed-state direction probe completed: ${OUTPUT_JSONL}"
