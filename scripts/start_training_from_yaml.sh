#!/bin/bash

conda activate llama-factory


EXT=maskedAll
DATA=debates
# DATA=sotu
# DATA=campaign

# MOD_DIR=Gemma-2b
MOD_DIR=Phi_1-5
MODEL_NAME="phi1-5b"  #"gemma2b"  # phi1-5b
# N_EPOCHS=15

# PROJ_DIR=/data/karen/debate-divisiveness/code/LM/revision/$DATA"_"$EXT
LF_DIR=/data/karen/LLaMA-Factory
# mkdir -p $PROJ_DIR

# OUT_DIR=$LF_DIR/saves/$DATA"_"$EXT/$MOD_DIR/train_2024-05-13
OUT_DIR=$LF_DIR/saves/$DATA"_"$EXT/$MOD_DIR/train_2024-05-13_15eps
DATA_DIR=$LF_DIR/data
mkdir -p $OUT_DIR

YAML_FILE=/data/karen/LLaMA-Factory/examples/kz_div_train/$MODEL_NAME"_lora_"$DATA.yaml  # "-"$N_EPOCHS

echo $YAML_FILE

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train $YAML_FILE