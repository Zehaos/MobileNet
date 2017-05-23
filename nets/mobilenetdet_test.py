from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets.mobilenetdet import *
from configs.kitti_config import config
import numpy as np


class MobileNetDetTest(tf.test.TestCase):
  def test_xywh_to_yxyx(self):
    with self.test_session() as sess:
      bbox = tf.constant([1, 2, 3, 4], dtype=tf.float32)
      bbox_yxyx = xywh_to_yxyx(bbox)
      output = sess.run(bbox_yxyx)
      self.assertAllEqual(output, [0, -0.5, 4, 2.5])
      bbox = tf.constant([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=tf.float32)
      bbox_yxyx = xywh_to_yxyx(bbox)
      output = sess.run(bbox_yxyx)
      self.assertAllEqual(output, [[0, -0.5, 4, 2.5], [0, -0.5, 4, 2.5]])

  def test_yxyx_to_xywh(self):
    with self.test_session() as sess:
      bbox = tf.constant([1, 2, 3, 4], dtype=tf.float32)
      bbox_xywh = yxyx_to_xywh(bbox)
      output = sess.run(bbox_xywh)
      self.assertAllEqual(output, [3, 2, 2, 2])

  def test_scale_bbox(self):
    with self.test_session() as sess:
      bbox = tf.constant([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=tf.float32)
      scaled_bbox = scale_bboxes(bbox, [10., 10.])
      output = sess.run(scaled_bbox)
      print(output)

  def test_iou(self):
    with self.test_session() as sess:
      bbox_1 = tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)
      bbox_2 = tf.constant([0.15, 0.15, 0.25, 0.25], dtype=tf.float32)
      iou_ = iou(bbox_1, bbox_2)
      output = sess.run(iou_)
      self.assertLess(np.abs(output - 1 / 7.), 1e-4)

  def test_compute_delta(self):
    with self.test_session() as sess:
      image_shape = [config.IMG_HEIGHT, config.IMG_WIDTH]
      fea_shape = [3, 3]
      anchors = set_anchors(image_shape, fea_shape)
      gt_box = tf.convert_to_tensor([0.25, 0.25, 0.0805153, 0.26666668])
      delta = compute_delta(gt_box, anchors[0][0][0])
      print(sess.run(delta))

  def test_batch_iou(self):
    with self.test_session() as sess:
      bboxes = tf.stack(
        [tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)] * 3
      )
      bbox = tf.constant([0.15, 0.15, 0.25, 0.25], dtype=tf.float32)
      iou = batch_iou(bboxes, bbox)
      output = sess.run(iou)
      self.assertTrue((np.abs(output - 1 / 7.) < 1e-4).all())

  def test_encode_annos(self):
    with self.test_session() as sess:
      batch_size = 2
      num_obj = 1
      image_shape = [90, 90]
      fea_shape = [3, 3]
      num_classes = config.NUM_CLASSES
      images = tf.constant(0, shape=[batch_size, image_shape[0], image_shape[1], 3])
      labels = tf.constant(1, shape=[batch_size, num_obj])
      anchors = set_anchors(image_shape, fea_shape)

      # Construct test bbox
      bbox_1 = tf.convert_to_tensor(xywh_to_yxyx(anchors[0][0][0]), dtype=tf.float32)
      bbox_2 = tf.convert_to_tensor(xywh_to_yxyx(anchors[2][2][1]), dtype=tf.float32)
      bboxes = tf.stack([tf.expand_dims(bbox_1, axis=0), tf.expand_dims(bbox_2, axis=0)])

      input_mask, labels_input, box_delta_input, box_input, anchors = \
        encode_annos(images, labels, bboxes, anchors, num_classes)
      out_input_mask, out_labels_input, out_box_delta_input, out_box_input, out_anchors = \
        sess.run([input_mask, labels_input, box_delta_input, box_input, anchors])
      print("reshape anchors", out_anchors)
      print("input_mask:", out_input_mask)
      print("box_input:", out_box_input)
      print("label_input:", out_labels_input)
      print("box_delta_input:", out_box_delta_input)
      print("shape:",
            "input_mask:", np.shape(out_input_mask),
            "labels_input:", np.shape(out_labels_input),
            "box_delta_input:", np.shape(out_box_delta_input),
            "box_input:", np.shape(out_box_input)
            )

  def test_set_anchors(self):
    with self.test_session() as sess:
      image_shape = [90, 90]
      fea_shape = [3, 3]
      anchors = set_anchors(image_shape, fea_shape)
      output = sess.run(anchors)
      self.assertAllEqual(np.shape(output), [fea_shape[0], fea_shape[1], config.NUM_ANCHORS, 4])
      print("Anchors:", output)
