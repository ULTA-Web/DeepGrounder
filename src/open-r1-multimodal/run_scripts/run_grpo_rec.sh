#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT/src/open-r1-multimodal"

: "${MODEL_PATH:?Set MODEL_PATH to the merged Qwen2.5-VL SFT checkpoint}"
: "${IMAGE_ROOT:?Set IMAGE_ROOT to DIOR_RSVG/JPEGImages}"
: "${DATA_CONFIG:?Set DATA_CONFIG to the absolute path of the GRPO dataset YAML}"

# Semantic verification is part of the full method. Never silently omit it.
REWARD_FUNCS="${REWARD_FUNCS:-accuracy format semantic}"
if [[ " $REWARD_FUNCS " == *" semantic "* ]]; then
    : "${DEEPGROUNDER_SEMANTIC_URL:?Set the Git-RSCLIP service URL; see docs/TRAINING.md}"
fi
RUN_NAME="${RUN_NAME:-DeepGrounder-DIOR-RSVG}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$RUN_NAME}"

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-12346}" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATA_CONFIG" \
    --image_root "$IMAGE_ROOT" \
    --reward_funcs $REWARD_FUNCS \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_prompt_length 1024 --max_completion_length 256 \
    --learning_rate 1e-6 --beta 0.04 --epsilon 0.2 --temperature 0.9 \
    --logging_steps 1 --bf16 --torch_dtype bfloat16 \
    --seed 42 --data_seed 42 --report_to none \
    --gradient_checkpointing true --attn_implementation flash_attention_2 \
    --num_train_epochs 2 --run_name "$RUN_NAME" \
    --save_steps 1000 --save_only_model true --freeze_vision_modules false \
    "$@"
