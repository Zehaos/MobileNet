#!/usr/bin/env bash

python tf_convert_data.py \
    --voc_root=/media/zehao/WD/Dataset/processed/VOC/VOCdevkit \
    --year=0712 \
    --split=trainval \
    --output_dir=/home/zehao/pascalvoc0712_tfrecord