#!/bin/bash


# Define hyperparameters
TASK="sst2"
MODEL_NAME="textattack/bert-base-uncased-SST-2"
# MODEL_NAME="textattack/bert-base-uncased-CoLA"
BATCH_SIZE=128
LEARNING_RATE=5e-4
MAX_EPOCHS=20
LOG_STEPS=10
LOSS="dal"
ADAPTER="disable"
MASK=30

# Define the directory path for saving the model
CURRENT_TIME=$(date +"%Y%m%d")
DIRPATH="./models/${MODEL_NAME}/${TASK}_${LOSS}_${ADAPTER}_${BATCH_SIZE}_${LEARNING_RATE}_${MAX_EPOCHS}_${MASK}/${CURRENT_TIME}"

# Run the script with the specified hyperparameters
CUDA_LAUNCH_BLOCKING=1 python dala.py \
    --task $TASK \
    --model_name_or_path $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --num_epochs $MAX_EPOCHS \
    --log_interval $LOG_STEPS \
    --dirpath $DIRPATH \
    --mask $MASK
