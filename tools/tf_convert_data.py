# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Convert a dataset to TFRecords format, which can be easily integrated into
a TensorFlow pipeline.

Usage:
```shell
python tf_convert_data.py \
    --datset_root=VOCdevkit \
    --year=0712 \
    --split=trainval \
    --output_dir=/tmp/pascalvoc0712_tfrecord
```
"""
import tensorflow as tf
import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../'))

from datasets import pascalvoc_to_tfrecords, kitti_object_to_tfrecords

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc',
    'The name of the dataset to convert.')
tf.app.flags.DEFINE_string(
    'dataset_root', None,
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'year', '0712',
    'Year of VOC dataset.')
tf.app.flags.DEFINE_string(
    'split', 'trainval',
    'Split of dataset, trainval/train/val/test.')
tf.app.flags.DEFINE_string(
    'output_dir', './',
    'Output directory where to store TFRecords files.')


def main(_):
    print('Dataset root dir:', FLAGS.dataset_root)
    print('Output directory:', FLAGS.output_dir)

    if FLAGS.dataset_name == 'pascalvoc':
        pascalvoc_to_tfrecords.run(FLAGS.dataset_root,
                                   FLAGS.year,
                                   FLAGS.split,
                                   FLAGS.output_dir,
                                   shuffling=True)
    elif FLAGS.dataset_name == 'kitti':
        kitti_object_to_tfrecords.run(FLAGS.dataset_root,
                                   FLAGS.split,
                                   FLAGS.output_dir,
                                   shuffling=True)
    else:
        raise ValueError('Dataset [%s] was not recognized.' % FLAGS.dataset_name)

if __name__ == '__main__':
    tf.app.run()

