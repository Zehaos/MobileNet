#!/bin/bash
#
# This script performs the following operations:
# 1. Trains a MobileNet model on the Imagenet training set.
# 2. Evaluates the model on the Imagenet validation set.
#
# Usage:
# ./scripts/train_mobilenetdet_on_kitti.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/mobilenetdet-model

# Where the dataset is saved to.
DATASET_DIR=/home/zehao/PycharmProjects/MobileNet/data/KITTI

# Where the checkpoint file stored.
CHECK_POINT=/home/zehao/PycharmProjects/MobileNet/model

# Run training.
python train_object_detector.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=kitti \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${CHECK_POINT} \
  --ignore_missing_vars=True \
  --model_name=mobilenetdet \
  --preprocessing_name=mobilenetdet \
  --width_multiplier=1.0 \
  --max_number_of_steps=1000000 \
  --batch_size=1 \
  --save_interval_secs=240 \
  --save_summaries_secs=240 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --rmsprop_decay=0.9 \
  --opt_epsilon=1.0\
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --momentum=0.9 \
  --num_epochs_per_decay=30.0 \
  --weight_decay=0.0 \
  --num_clones=1