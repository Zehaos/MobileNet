from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from preprocessing import preprocessing_factory
from configs.kitti_config import config
from nets.mobilenetdet import scale_bboxes

from datasets import dataset_factory
from tensorflow.contrib import slim

dataset = dataset_factory.get_dataset(
  'kitti', 'train', '/home/zehao/Dataset/KITII/tfrecord')

# def conver_box(bboxes, img_h, img_w):
#   [ymin, xmin, ymax, xmax] = tf.unstack(bboxes, axis=1)
#   img_h = tf.cast(img_h, tf.float32)
#   img_w = tf.cast(img_w, tf.float32)
#   ymin = tf.truediv(ymin, img_h)
#   xmin = tf.truediv(xmin, img_w)
#   ymax = tf.truediv(ymax, img_h)
#   xmax = tf.truediv(xmax, img_w)
#   return tf.expand_dims(tf.stack([ymin,xmin,ymax,xmax], axis=1), axis=0)

with tf.Graph().as_default() as graph:
  with tf.device('/cpu:0'):
    provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=1,
      common_queue_capacity=20 * 1,
      common_queue_min=10 * 1)
    [image, shape, bbox, label] = provider.get(['image', 'shape', 'object/bbox', 'object/label'])

    bbox = scale_bboxes(bbox, shape)
    bbox = tf.expand_dims(bbox, axis=0)
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
          'mobilenetdet',
          is_training=True)

    image, gt_labels, gt_bboxes = image_preprocessing_fn(image,
                                                         config.IMG_HEIGHT,
                                                         config.IMG_WIDTH,
                                                         labels=label,
                                                         bboxes=bbox,
                                                         )



  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # [image, bbox, label] = sess.run([image, gt_bboxes, gt_labels])

    summary_writer = tf.summary.FileWriter("/home/zehao/PycharmProjects/MobileNet/summary", sess.graph)
    merge = tf.summary.merge_all()
    merge = sess.run(merge)
    summary_writer.add_summary(merge)
    summary_writer.close()
