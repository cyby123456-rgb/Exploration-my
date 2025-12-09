#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
N_NODE=1
EXPERIMENT_NAME=qwen3-1.7b-2gpu-PPO-qwen2.5b
DATA=parquet/stage1/polaris-data-53K.parquet

NOISE_STD=0.0
NOISE_LAYER_IDX=null
NOISE_PHASE=eval
VAL_NOISE_STD=0.0
ENTROPY_MODE=token
VAL_NOISE_LAYER_IDX=25
VAL_NOISE_ALL_LAYERS=false
VAL_NOISE_DECAY_STEPS=0.0
VAL_NOISE_DECAY_MIN_STD=0.0
CRITIC_DISTRIBUTIONAL=false
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
    data.train_batch_size=152 \
    data.val_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/liujinyi/models_hf/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562  \
    critic.model.path=/mnt/shared-storage-user/liujinyi/models_hf/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
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
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
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
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Polaris-Reproduce-1.7B-v2' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=$N_NODE \
    trainer.debug=False \
    trainer.dyn_sampling_polaris=False \
    trainer.save_freq=500 \
    trainer.test_freq=25 \
    ++trainer.validation_noise.std=${VAL_NOISE_STD} \
    ++trainer.validation_noise.layer_idx=${VAL_NOISE_LAYER_IDX} \
    ++trainer.validation_noise.all_layers=${VAL_NOISE_ALL_LAYERS} \
    ++trainer.validation_noise.decay.steps=${VAL_NOISE_DECAY_STEPS} \
    ++trainer.validation_noise.decay.min_std=${VAL_NOISE_DECAY_MIN_STD} \
    ++critic.distributional=${CRITIC_DISTRIBUTIONAL} \
    ++critic.num_quantiles=${CRITIC_NUM_QUANTILES} \
    ++critic.quantile_huber_kappa=${CRITIC_QUANTILE_KAPPA} \
    ++critic.quantile_mode=${CRITIC_QUANTILE_MODE} \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=15 "${@:1}"



# Echo the command for verification
echo "Command to be executed:"
echo "$CMD"

# Execute the command
eval "$CMD"
