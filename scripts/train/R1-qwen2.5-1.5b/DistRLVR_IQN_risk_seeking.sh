#!/bin/bash
set -e
set -x

# ================= Environment init (edit as needed) =================
export PATH="/root/miniconda3/bin:$PATH"

CONDA_ENV="${CONDA_ENV:-test123}"
REPO_DIR="${REPO_DIR:-/mnt/shared-storage-user/liujinyi/test123/POLARIS-main/POLARIS-main}"
if command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_ENV}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi
cd "${REPO_DIR}"

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
N_NODE=1
PROJECT_NAME="${PROJECT_NAME:-Qwen2.5-1.5B-Distributional-v3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-DistRLVR_IQN_risk_seeking}"
DATA=parquet/stage1/polaris-data-53K.parquet

# ================= CUDA (train) =================
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"

# ================= Checkpoint / merge / eval params =================
CKPT_ROOT="${CKPT_ROOT:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${REPO_DIR}/merged_models/${EXPERIMENT_NAME}}"

# Eval settings
EVAL_EXPERIMENT_NAME="${EVAL_EXPERIMENT_NAME:-${EXPERIMENT_NAME}-eval}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results/${EVAL_EXPERIMENT_NAME}}"

# Inference params
TEMP=1.0
MAX_LEN=1024
TOP_K=-1
N_SAMPLES=128
BATCH_SIZE=1024
MAX_BATCH_TOKENS=6144
N_GPUS=2
TENSOR_MODEL_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9

# Tracking
TRACKING_PROJECT="${TRACKING_PROJECT:-${EVAL_EXPERIMENT_NAME}}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
WANDB_SYNC_ON_EXIT="${WANDB_SYNC_ON_EXIT:-0}"

# ================= Dataset paths (edit here) =================
DATASET_PATHS=(
    "evaluation/benchmarks/aime24.parquet"
    "evaluation/benchmarks/aime25.parquet"
    "evaluation/benchmarks/amc23.parquet"
    "evaluation/benchmarks/minerva.parquet"
    "evaluation/benchmarks/olympiad.parquet"
    "evaluation/benchmarks/deepscaler/math.parquet"
)

NOISE_STD=0.0
NOISE_LAYER_IDX=null
NOISE_PHASE=eval
VAL_NOISE_STD=0.0
ENTROPY_MODE=token
VAL_NOISE_LAYER_IDX=25
VAL_NOISE_ALL_LAYERS=false
VAL_NOISE_DECAY_STEPS=0.0
VAL_NOISE_DECAY_MIN_STD=0.0
CRITIC_DISTRIBUTIONAL=true
CRITIC_DISTRIBUTIONAL_V2=false
CRITIC_DISTRIBUTIONAL_V3=true
CRITIC_NUM_QUANTILES=32
CRITIC_QUANTILE_KAPPA=1.0
CRITIC_QUANTILE_MODE=iqn
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_path)
            DATA="$2"
            shift 2
            ;;
        --n_node)
            N_NODE="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --noise_std)
            NOISE_STD="$2"
            shift 2 
            ;;
        --noise_layer_idx)
            NOISE_LAYER_IDX="$2"
            shift 2 
            ;;
        --noise_phase)
            NOISE_PHASE="$2"
            shift 2 
            ;;
        --entropy_mode)
            ENTROPY_MODE="$2"
            shift 2 
            ;;
        --val_noise_std)
            VAL_NOISE_STD="$2"
            shift 2
            ;;
        --val_noise_layer_idx)
            VAL_NOISE_LAYER_IDX="$2"
            shift 2
            ;;
        --entropy_mode)
            ENTROPY_MODE="$2"
            shift 2
            ;;
        --critic_distributional)
            CRITIC_DISTRIBUTIONAL="$2"
            shift 2
            ;;
        --critic_distributional_v2)
            CRITIC_DISTRIBUTIONAL_V2="$2"
            shift 2
            ;;
        --critic_distributional_v3)
            CRITIC_DISTRIBUTIONAL_V3="$2"
            shift 2
            ;;
        --critic_num_quantiles)
            CRITIC_NUM_QUANTILES="$2"
            shift 2
            ;;
        --critic_quantile_kappa)
            CRITIC_QUANTILE_KAPPA="$2"
            shift 2
            ;;
        --critic_quantile_mode)
            CRITIC_QUANTILE_MODE="$2"
            shift 2
            ;;
        --val_noise_all_layers)
            VAL_NOISE_ALL_LAYERS="$2"
            shift 2
            ;;
        --val_noise_decay_steps)
            VAL_NOISE_DECAY_STEPS="$2"
            shift 2
            ;;
        --val_noise_min_std)
            VAL_NOISE_DECAY_MIN_STD="$2"
            shift 2
            ;;

        *)
            break
            ;;
    esac
done


# Train over a single node, 8 H800-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=/mnt/shared-storage-user/liujinyi/test123/POLARIS-main/POLARIS-main/evaluation/benchmarks/deepscaler/train.parquet \
    data.val_files=/mnt/shared-storage-user/liujinyi/test123/POLARIS-main/POLARIS-main/evaluation/benchmarks/deepscaler/math.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=8 \
    data.max_prompt_length=766 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/liujinyi/models_hf/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562  \
    critic.model.path=/mnt/shared-storage-user/liujinyi/models_hf/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    ++actor_rollout_ref.model.override_config.hidden_noise_std=${NOISE_STD} \
    ++actor_rollout_ref.model.override_config.hidden_noise_layer_idx=${NOISE_LAYER_IDX} \
    ++actor_rollout_ref.model.override_config.hidden_noise_phase=${NOISE_PHASE} \
    ++actor_rollout_ref.actor.entropy_mode=${ENTROPY_MODE} \
    algorithm.kl_ctrl.kl_coef=0.001 \
    ++critic.risk_apply_to=baseline \
    ++critic.risk_level=cvar_upper_0.25 \
    ++algorithm.risk_apply_to=baseline1 \
    ++algorithm.risk_level=cvar_upper_0.25 \
    ++algorithm.baseline_mode=risk \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Qwen2.5-1.5B-Distributional-v2' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=$N_NODE \
    trainer.debug=False \
    trainer.dyn_sampling_polaris=False \
    trainer.save_freq=314 \
    trainer.test_freq=25 \
    ++actor_rollout_ref.seed=42 \
    ++data.seed=42 \
    ++trainer.validation_noise.std=${VAL_NOISE_STD} \
    ++trainer.validation_noise.layer_idx=${VAL_NOISE_LAYER_IDX} \
    ++trainer.validation_noise.all_layers=${VAL_NOISE_ALL_LAYERS} \
    ++trainer.validation_noise.decay.steps=${VAL_NOISE_DECAY_STEPS} \
    ++trainer.validation_noise.decay.min_std=${VAL_NOISE_DECAY_MIN_STD} \
    ++critic.distributional=${CRITIC_DISTRIBUTIONAL} \
    ++critic.distributional_v2=${CRITIC_DISTRIBUTIONAL_V2} \
    ++critic.distributional_v3=${CRITIC_DISTRIBUTIONAL_V3} \
    ++critic.num_quantiles=${CRITIC_NUM_QUANTILES} \
    ++critic.quantile_huber_kappa=${CRITIC_QUANTILE_KAPPA} \
    ++critic.quantile_mode=${CRITIC_QUANTILE_MODE} \
    ++critic.use_action_response_mask=False \
    critic.model.use_remove_padding=True \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 "${@:1}"

# ================= Find latest checkpoint =================
LATEST_STEP_FILE="${CKPT_ROOT}/latest_checkpointed_iteration.txt"
if [[ -f "${LATEST_STEP_FILE}" ]]; then
    LATEST_STEP="$(cat "${LATEST_STEP_FILE}")"
    ACTOR_CKPT_DIR="${CKPT_ROOT}/global_step_${LATEST_STEP}/actor"
else
    LATEST_STEP_DIR="$(ls -d "${CKPT_ROOT}"/global_step_* 2>/dev/null | sort -V | tail -n 1)"
    if [[ -z "${LATEST_STEP_DIR}" ]]; then
        echo "No checkpoint found under ${CKPT_ROOT}" >&2
        exit 1
    fi
    ACTOR_CKPT_DIR="${LATEST_STEP_DIR}/actor"
fi

echo "Using checkpoint: ${ACTOR_CKPT_DIR}"

# ================= Merge checkpoint into HF model =================
python3 verl/scripts/model_merger.py \
    --backend fsdp \
    --local_dir "${ACTOR_CKPT_DIR}" \
    --target_dir "${MERGED_MODEL_DIR}"

# ================= CUDA / Ray / vLLM (eval) =================
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=2
export RAYON_NUM_THREADS=2
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION_V2
export WANDB_MODE=offline

# ================= Create output dir =================
mkdir -p "${OUTPUT_DIR}"

echo "======================================"
echo "Eval experiment: ${EVAL_EXPERIMENT_NAME}"
echo "Model: ${MERGED_MODEL_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Datasets: ${DATASET_PATHS[@]}"
echo "======================================"

# ================= Run generation + eval =================
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    DATASET_TAG="$(basename "${DATASET_PATH}" .parquet)"
    OUTPUT_PATH="${OUTPUT_DIR}/${DATASET_TAG}-${TEMP}-${N_SAMPLES}-${MAX_LEN}-${TOP_K}.parquet"

    echo "--------------------------------------"
    echo "Dataset: ${DATASET_TAG}"
    echo "Input:  ${DATASET_PATH}"
    echo "Output: ${OUTPUT_PATH}"
    echo "--------------------------------------"

    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="${N_GPUS}" \
        data.path="${DATASET_PATH}" \
        data.output_path="${OUTPUT_PATH}" \
        data.n_samples="${N_SAMPLES}" \
        data.batch_size="${BATCH_SIZE}" \
        model.path="${MERGED_MODEL_DIR}" \
        rollout.temperature="${TEMP}" \
        rollout.enable_chunked_prefill=True \
        rollout.max_num_batched_tokens="${MAX_BATCH_TOKENS}" \
        rollout.response_length="${MAX_LEN}" \
        rollout.top_k="${TOP_K}" \
        rollout.top_p=1.0 \
        rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
        rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}"

    python3 -m verl.trainer.main_eval \
        data.path="${OUTPUT_PATH}" \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model \
        tracking.project_name="${TRACKING_PROJECT}" \
        tracking.experiment_name="${EVAL_EXPERIMENT_NAME}-${DATASET_TAG}" \
        tracking.logger=['console','wandb']
done

echo "All evaluations finished successfully!"

if [[ "${WANDB_MODE}" == "offline" && "${WANDB_SYNC_ON_EXIT}" == "1" ]]; then
    wandb sync "${WANDB_DIR}"
fi
