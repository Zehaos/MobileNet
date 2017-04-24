# Copyright 2015 Zehao Shi. All Rights Reserved.
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
"""Converts KITTI data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.pascalvoc_common import VOC_LABELS

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'label_2/'
DIRECTORY_IMAGES = 'image_2/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

CLASSES = {
    'Pedestrian': 1,
    'Cyclist': 2,
    'Car': 3,
    'Person_sitting': 4,
    'Van': 5,
    'Tram': 6,
    'Truck': 7,
    'Misc': 8,
    'DontCare': 9
}


def _process_image(directory, name):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = os.path.join(directory, DIRECTORY_IMAGES, name + '.png')
    image_data = tf.gfile.FastGFile(filename, 'r').read()
    img_tensor = tf.image.decode_png(image_data)
    shape = tf.shape(img_tensor).aslist()

    # Read the txt annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.txt')
    with open(filename) as anno_file:
      objects = anno_file.readlines()

    label_list = []
    type_list = []
    trun_list = []
    occl_list = []
    alpha_list = []
    bbox_x1_list = []
    bbox_y1_list = []
    bbox_x2_list = []
    bbox_y2_list = []
    ddd_bbox_h_list = []
    ddd_bbox_w_list = []
    ddd_bbox_l_list = []
    ddd_bbox_x_list = []
    ddd_bbox_y_list = []
    ddd_bbox_z_list = []
    ddd_bbox_ry_list = []

    for object in objects:
        obj_anno = object.split(' ')
        type_txt = obj_anno[0].encode('ascii')
        truncation = int(obj_anno[1])  # [0..1] truncated pixel ratio
        occlusion = int(obj_anno[2])  # 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
        alpha = float(obj_anno[3])  # object observation angle([-pi..pi])

        label_list.append(CLASSES[type_txt])
        type_list.append(type_txt)
        trun_list.append(truncation)
        occl_list.append(occlusion)
        alpha_list.append(alpha)

        # Bounding Box
        bbox_x1 = float(obj_anno[4])
        bbox_y1 = float(obj_anno[5])
        bbox_x2 = float(obj_anno[6])
        bbox_y2 = float(obj_anno[7])
        bbox_x1_list.append(bbox_x1)
        bbox_y1_list.append(bbox_y1)
        bbox_x2_list.append(bbox_x2)
        bbox_y2_list.append(bbox_y2)

        # 3D bounding box
        ddd_bbox_h = float(obj_anno[8])
        ddd_bbox_w = float(obj_anno[9])
        ddd_bbox_l = float(obj_anno[10])
        ddd_bbox_x = float(obj_anno[11])
        ddd_bbox_y = float(obj_anno[12])
        ddd_bbox_z = float(obj_anno[13])
        ddd_bbox_ry = float(obj_anno[14])
        ddd_bbox_h_list.append(ddd_bbox_h)
        ddd_bbox_w_list.append(ddd_bbox_w)
        ddd_bbox_l_list.append(ddd_bbox_l)
        ddd_bbox_x_list.append(ddd_bbox_x)
        ddd_bbox_y_list.append(ddd_bbox_y)
        ddd_bbox_z_list.append(ddd_bbox_z)
        ddd_bbox_ry_list.append(ddd_bbox_ry)

    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
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
            'image/object/bbox/occlusion': int64_feature(occl_list),
            'image/object/bbox/truncation': int64_feature(trun_list),
            'image/object/observation/alpha': float_feature(alpha_list),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data),
            'image/object/3Dbbox/h': float_feature(ddd_bbox_h_list),
            'image/object/3Dbbox/w': float_feature(ddd_bbox_w_list),
            'image/object/3Dbbox/l': float_feature(ddd_bbox_l_list),
            'image/object/3Dbbox/x': float_feature(ddd_bbox_x_list),
            'image/object/3Dbbox/y': float_feature(ddd_bbox_y_list),
            'image/object/3Dbbox/z': float_feature(ddd_bbox_z_list),
            'image/object/3Dbbox/ry': float_feature(ddd_bbox_ry_list)
    }))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    example = _process_image(dataset_dir, name)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(kitti_root, split, output_dir, shuffling=False):
    """Runs the conversion operation.

    Args:
      voc_root: KITTI dataset root dir.
      split: train/test
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
    image_dir = os.path.join(kitti_root, '%sing'%split, 'image_2')
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, split, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i].strip()
                _add_to_tfrecord(image_dir, filename, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    print('\nFinished converting the Pascal VOC dataset!')
