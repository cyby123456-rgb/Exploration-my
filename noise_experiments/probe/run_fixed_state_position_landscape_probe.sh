#!/usr/bin/env bash
# Self-contained rjob launcher for the multi-position fixed-state probe.
set -euo pipefail
set -x

REPO_ROOT="/mnt/shared-storage-user/liujinyi/test123/Noise-Experiment"
VERL_ROOT="${REPO_ROOT}/verl"
CONDA_BASE="/root/miniconda3"
CONDA_ENV="test123"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

# ================= Experiment (edit here) =================
EXPERIMENT_NAME="r1-1p5b-32-8k-grpo-clean-seed1002"
DATASET_TAG="aime24"
MODEL_PATH="${REPO_ROOT}/model_merged/grpo-8k"
CLEAN_PARQUET="${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}/${DATASET_TAG}-1.0-32-8192--1.parquet"

LAYER=24
BASE_OUTCOME="wrong" # wrong: W->R; correct: R->W (run separately)
POSITION_FRACTIONS="0.25,0.5,0.75"
ALPHAS="0.025,0.05"
DIRECTIONS=32
DIRECTION_BATCH_SIZE=8
MAX_STATES=12
# Keep all questions which have at least one wrong sample.  The requested
# base_outcome filter below removes all-correct questions for a W->R probe.
MIN_CLEAN_CORRECT=0
MAX_CLEAN_CORRECT=31
MAX_NEW_TOKENS=2048
MAX_INPUT_TOKENS=8192
PROBE_SEED=20260810
DTYPE="bfloat16"

OUTPUT_DIR="${REPO_ROOT}/evaluation/results/${EXPERIMENT_NAME}/position_landscape"
OUTPUT_JSONL="${OUTPUT_DIR}/${DATASET_TAG}-layer${LAYER}-${BASE_OUTCOME}-seed${PROBE_SEED}.jsonl"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${CLEAN_PARQUET}" ]]; then
    echo "Clean generation parquet not found: ${CLEAN_PARQUET}" >&2
    exit 1
fi

python3 "${REPO_ROOT}/noise_experiments/probe/run_fixed_state_position_landscape_probe.py" \
    --input-parquet "${CLEAN_PARQUET}" \
    --model "${MODEL_PATH}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --layer "${LAYER}" \
    --base-outcome "${BASE_OUTCOME}" \
    --position-fractions "${POSITION_FRACTIONS}" \
    --alphas "${ALPHAS}" \
    --directions "${DIRECTIONS}" \
    --direction-batch-size "${DIRECTION_BATCH_SIZE}" \
    --max-states "${MAX_STATES}" \
    --min-clean-correct "${MIN_CLEAN_CORRECT}" \
    --max-clean-correct "${MAX_CLEAN_CORRECT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --seed "${PROBE_SEED}" \
    --dtype "${DTYPE}"

python3 "${REPO_ROOT}/noise_experiments/probe/analyze_direction_landscape.py" \
    --input-jsonl "${OUTPUT_JSONL}"
