# Copyright 2017 Zehao Shi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts KITTI Object data to TFRecords file format with Example protos.

The raw KITTI Object data set is expected to reside in PNG files located in the
directory 'image_2'. Similarly, annotations are supposed to be stored in the
'label_2'.

"""
import os
import sys
import random
import re

import tensorflow as tf
import cv2
import numpy as np
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

CLASSES = {
    'Pedestrian': 0,
    'Cyclist': 1,
    'Car': 2,
}

def _process_image(directory, split, name):
    # Read the image file.
    filename = os.path.join(directory, 'image_2', name + '.png')
    image_data = tf.gfile.FastGFile(filename, 'r').read()

    # Get shape
    img = cv2.imread(filename)
    shape = np.shape(img)

    label_list = []
    type_list = []

    bbox_x1_list = []
    bbox_y1_list = []
    bbox_x2_list = []
    bbox_y2_list = []


    # If 'test' split, skip annotations
    if re.findall(r'train', split):
      # Read the txt annotation file.
      filename = os.path.join(directory, 'label_2', name + '.txt')
      with open(filename) as anno_file:
        objects = anno_file.readlines()

      for object in objects:
          obj_anno = object.split(' ')
          type_txt = obj_anno[0].encode('ascii')
          if type_txt in CLASSES:
            label_list.append(CLASSES[type_txt])
            type_list.append(type_txt)

            # Bounding Box
            bbox_x1 = float(obj_anno[4])
            bbox_y1 = float(obj_anno[5])
            bbox_x2 = float(obj_anno[6])
            bbox_y2 = float(obj_anno[7])
            bbox_x1_list.append(bbox_x1)
            bbox_y1_list.append(bbox_y1)
            bbox_x2_list.append(bbox_x2)
            bbox_y2_list.append(bbox_y2)

    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(image_data),
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(bbox_x1_list),
            'image/object/bbox/xmax': float_feature(bbox_x2_list),
            'image/object/bbox/ymin': float_feature(bbox_y1_list),
            'image/object/bbox/ymax': float_feature(bbox_y2_list),
            'image/object/bbox/label': int64_feature(label_list),
            'image/object/bbox/label_text': bytes_feature(type_list),
    }))
    return example


def _add_to_tfrecord(dataset_dir, split, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      split: train/val/test
      name: Image name;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    example = _process_image(dataset_dir, split, name)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(kitti_root, split, output_dir, shuffling=False):
    """Runs the conversion operation.

    Args:
      kitti_root: KITTI dataset root dir.
      split: trainval/train/val
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    # Dataset filenames, and shuffling.
    split_file_path = os.path.join(kitti_root,
                                 'ImageSets',
                                 '%s.txt'%split)
    with open(split_file_path) as f:
        filenames = f.readlines()

    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    image_dir = os.path.join(kitti_root, '%sing'%split)
    if split == 'val':
      image_dir = os.path.join(kitti_root, '%sing' % 'train')
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, split, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i].strip()
                _add_to_tfrecord(image_dir, split, filename, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    print('\nFinished converting the KITTI dataset!')
