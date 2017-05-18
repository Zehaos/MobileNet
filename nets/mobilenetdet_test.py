from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets.mobilenetdet import *
import numpy as np

slim = tf.contrib.slim

class MobileNetDetTest(tf.test.TestCase):
  def test_xywh_to_yxyx(self):
    with self.test_session() as sess:
      bbox = tf.constant([1,2,3,4], dtype=tf.float32)
      bbox_yxyx = xywh_to_yxyx(bbox)
      output = sess.run(bbox_yxyx)
      self.assertAllEqual(output, [0, -0.5, 4, 2.5])

  def test_yxyx_to_xywh(self):
    with self.test_session() as sess:
      bbox = tf.constant([1,2,3,4], dtype=tf.float32)
      bbox_xywh = yxyx_to_xywh(bbox)
      output = sess.run(bbox_xywh)
      self.assertAllEqual(output, [3, 2, 2, 2])

  def test_iou(self):
    with self.test_session() as sess:
      bbox_1 = tf.constant([1,1,2,2], dtype=tf.float32)
      bbox_2 = tf.constant([1.5, 1.5, 2.5, 2.5], dtype=tf.float32)
      iou_ = iou(bbox_1, bbox_2)
      output = sess.run(iou_)
      self.assertLess(np.abs(output-1/7.), 1e-4)

  def test_batch_iou(self):
    with self.test_session() as sess:
      bboxes = tf.stack(
        [tf.constant([1,1,2,2], dtype=tf.float32)]*3
      )
      bbox = tf.constant([1.5, 1.5, 2.5, 2.5], dtype=tf.float32)
      iou = batch_iou(bboxes, bbox)
      output = sess.run(iou)
      self.assertTrue((np.abs(output - 1 / 7.)<1e-4).all())

  def test_encode_annos(self):
    with self.test_session() as sess:
      batch_size = 10
      num_obj = 10
      image_shape = [500,500]
      fea_shape = [10, 10]
      num_classes = 21
      images = tf.constant(0, shape=[batch_size, image_shape[0], image_shape[1], 3])
      labels = tf.constant(1, shape=[batch_size, num_obj])
      bboxes = tf.constant(2, shape=[batch_size, num_obj, 4], dtype=tf.float32)
      anchors = set_anchors(image_shape, fea_shape)
      input_mask, labels_input, box_delta_input, box_input = \
        encode_annos(images, labels, bboxes, anchors, num_classes)
      out_input_mask, out_labels_input, out_box_delta_input, out_box_input = \
        sess.run([input_mask, labels_input, box_delta_input, box_input])
      print("shape:",
            np.shape(out_input_mask),
            np.shape(out_labels_input),
            np.shape(out_box_delta_input),
            np.shape(out_box_input)
            )

  def test_set_anchors(self):
    with self.test_session() as sess:
      image_shape = [500,500]
      fea_shape = [10, 10]
      anchors = set_anchors(image_shape, fea_shape)
      output = sess.run(anchors)
      self.assertAllEqual(np.shape(output), [10, 10, 9, 4])