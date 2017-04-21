#!/bin/bash
#
# This script performs the following operations:
# 1. Trains a MobileNet model on the Imagenet training set.
# 2. Evaluates the model on the Imagenet testing set.
#
# Usage:
# ./scripts/train_mobilenet_on_imagenet.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/mobilenet-model

# Where the dataset is saved to.
DATASET_DIR=/home/zehao/Dataset/imagenet-data

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=imagenet \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet \
  --preprocessing_name=mobilenet \
  --width_multiplier=1.0 \
  --max_number_of_steps=100000 \
  --batch_size=1 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

# Run evaluation.
#python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR} \
#  --eval_dir=${TRAIN_DIR} \
#  --dataset_name=imagenet \
#  --dataset_split_name=test \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=mobilenet
