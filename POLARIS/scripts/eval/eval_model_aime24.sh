#!/bin/bash

set -x

unset VLLM_ATTENTION_BACKEND

NAME="polaris-4b"
TEMP=1.4
L=90000
K=20
N=32
OUTPUT_DIR="evaluation/results"  # Add default output directory


while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --experiment_name)
            NAME="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --t)
            TEMP="$2"
            shift 2
            ;;
        --max_length)
            L="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
    esac
done
echo "MODEL_PATH: $MODEL_PATH"
echo "EXP NAME: $NAME"
echo "N: $N"
echo "Temperature: $TEMP"
echo "max-length: $L"
echo "top-K: $K"

# Echo the values for verification
echo "Output file: ${OUTPUT_DIR}/${NAME}/aime24-${TEMP}-${N}-${L}-${K}.parquet"

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=evaluation/benchmarks/aime24.parquet \
    data.output_path=${OUTPUT_DIR}/${NAME}/aime24-${TEMP}-${N}-${L}-${K}.parquet \
    data.n_samples=${N} \
    data.batch_size=102400 \
    model.path=${MODEL_PATH} \
    rollout.temperature=${TEMP} \
    rollout.enable_chunked_prefill=True \
    rollout.max_num_batched_tokens=96000 \
    rollout.response_length=${L} \
    rollout.top_k=${K} \
    rollout.top_p=1.0 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.tensor_model_parallel_size=1

