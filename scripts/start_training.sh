#!/bin/bash

conda activate llama-factory


EXT=maskedAll
# DATA=debates
# DATA=sotu
DATA=campaign

PROJ_DIR=/data/karen/debate-divisiveness/code/LM/revision/$DATA"_"$EXT
LF_DIR=/data/karen/LLaMA-Factory
# mkdir -p $PROJ_DIR

OUT_DIR=$LF_DIR/saves/$DATA"_"$EXT/Gemma-2b/train_2024-05-07
DATA_DIR=$LF_DIR/data
mkdir -p $OUT_DIR



CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage pt \
    --do_train True \
    --model_name_or_path google/gemma-2b \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --template default \
    --flash_attn auto \
    --dataset_dir data \
    --dataset $DATA"_"$EXT \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 10.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 500 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir $OUT_DIR \
    --fp16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --plot_loss True \
    --overwrite_output_dir True \
    --new_special_tokens "<DEBATE_START>,<DEBATE_END>,<ENT>" \
    --lr_scheduler_type linear  # to match gpt-2