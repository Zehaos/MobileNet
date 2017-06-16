#!/bin/bash
#
# This script performs the following operations:
# 1. Trains a MobileNet model on the kitti dataset.
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
  --batch_size=10 \
  --save_interval_secs=240 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=sgd \
  --learning_rate=0.00002 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=2.0 \
  --weight_decay=0.000001 \
  --num_clones=1

#--train_dir=/tmp/mobilenetdet-model
#--dataset_name=kitti
#--dataset_split_name=train
#--dataset_dir=/home/zehao/Dataset/KITII/tfrecord
#--checkpoint_path=/home/zehao/PycharmProjects/MobileNet/model
#--ignore_missing_vars=True
#--model_name=mobilenetdet
#--preprocessing_name=mobilenetdet
#--width_multiplier=1.0
#--max_number_of_steps=1000000
#--batch_size=3
#--save_interval_secs=240
#--save_summaries_secs=60
#--log_every_n_steps=1
#--optimizer=sgd
#--learning_rate=0.00005
#--learning_rate_decay_factor=0.5
#--num_epochs_per_decay=30.0
#--weight_decay=0.000001
#--num_clones=1
