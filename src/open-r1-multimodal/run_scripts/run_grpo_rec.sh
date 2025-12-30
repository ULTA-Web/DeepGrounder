cd src/open-r1-multimodal

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="LoRA8-GRPO-dIOU0.5-Semantic-DIOR"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

#    --image_root /mnt/dataset/zhangjiafan/train2014-COCO\
#    --image_root /mnt/dataset/zhangjiafan/RRSISD/data/images/JPEGImages\

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir /mnt/dataset/zhangjiafan/Checkpoints/DIOR_RSVG/$RUN_NAME \
    --model_name_or_path /mnt/dataset/zhangjiafan/Checkpoints/DIOR_RSVG/DIOR-LoRA8 \
    --dataset_name data_config/rec.yaml \
    --image_root /mnt/dataset/zhangjiafan/RRSISD/data/images/JPEGImages\
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model true



#        --model_name_or_path /mnt/dataset/zhangjiafan/Checkpoints/Qwen2.5-VL-3B_merge_lora8 \