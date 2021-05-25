#!/bin/bash
# MODEL_NAME='densenet169'
# MODEL_NAME='densenet161'
# MODEL_NAME='densenet121'
# MODEL_NAME='resnet152'
# MODEL_NAME='resnet101'
# MODEL_NAME='resnet50'
# MODEL_NAME='resnet101'
# MODEL_NAME='resnet152'
# MODEL_NAME='densenet121'
# MODEL_NAME='densenet161'
MODEL_NAME='densenet169'
# MODEL_NAME='densenet161'
# MODEL_NAME='densenet121'
# MODEL_NAME='resnet152'
# MODEL_NAME='resnet101'
# MODEL_NAME='resnet50'
NUM_EPOCHS=50
SAVE_MODEL_PATH="./results/all_models/latest/att_densenet169_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/att_densenet161_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/att_densenet121_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/att_resnet101_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/att_resnet50_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/fpn_resnet101_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/fpn_resnet152_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/fpn_densenet121_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/fpn_densenet161_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/fpn_densenet169_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/single_densenet169_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/single_densenet161_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/single_resnet152_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/single_resnet101_xiugao.pth"
# SAVE_MODEL_PATH="./results/all_models/latest/single_resnet50_xiugao.pth"
LOG_DIR="./logs/xiugao"
# LOG_DIR="./logs/xiugao_121"
# LOG_DIR="./logs/xiugao_101"
# LOG_DIR="./logs/xiugao_50"
# LOG_DIR="./logs/xiugao_fpn_101"
# LOG_DIR="./logs/xiugao_fpn_152"
# LOG_DIR="./logs/xiugao_fpn_161"
# LOG_DIR="./logs/xiugao_single_161"
# LOG_DIR="./logs/xiugao_single_121"
# LOG_DIR="./logs/xiugao_single_152"
# LOG_DIR="./logs/xiugao_single_101"
# LOG_DIR="./logs/xiugao_single_50"


python run.py \
    --model_name $MODEL_NAME \
    --num_epochs $NUM_EPOCHS \
    --save_model_path $SAVE_MODEL_PATH \
    --log_dir $LOG_DIR \
    SGD
