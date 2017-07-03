#!/bin/bash
#
# This script performs the following operations:
# 1. Trains a MobileNet model on the MNIST training set.
# 2. Evaluates the model on the MNIST validation set.
#
# Usage:
# ./scripts/train_mobilenet_on_mnist.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/lenet-model

# Where the dataset is saved to.
DATASET_DIR=/Users/Mohamad/Projects/Datasets/MNIST_data/

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet \
  --preprocessing_name=lenet \
  --batch_size=64 \
  --save_interval_secs=240 \
  --save_summaries_secs=240 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --rmsprop_decay=0.9 \
  --opt_epsilon=1.0\
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --momentum=0.9 \
  --num_epochs_per_decay=20.0 \
  --weight_decay=0.0 \
  --num_clones=2 \
  --clone_on_cpu=True

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet
