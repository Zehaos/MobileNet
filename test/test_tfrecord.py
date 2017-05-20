from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2

from datasets import dataset_factory
from tensorflow.contrib import slim

dataset = dataset_factory.get_dataset(
  'kitti', 'train', '/home/zehao/Dataset/KITII/tfrecord')

with tf.device('/cpu:0'):
  provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    num_readers=1,
    common_queue_capacity=20 * 1,
    common_queue_min=10 * 1)
  [image, shape, bbox, label] = provider.get(['image', 'shape', 'object/bbox', 'object/label'])

with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  [image, shape, bbox, label] = sess.run([image, shape, bbox, label])
  print(shape)
  print(bbox)
  print(label)
  image = image[:, :, ::-1]
  cv2.imshow('show', image)
  cv2.waitKey(0)
