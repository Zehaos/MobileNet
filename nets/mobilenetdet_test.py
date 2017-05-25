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
      anchors = set_anchors(img_shape=[config.IMG_HEIGHT, config.IMG_WIDTH],
                            fea_shape=[config.FEA_HEIGHT, config.FEA_WIDTH])
      anchors_shape = anchors.get_shape().as_list()
      fea_h = anchors_shape[0]
      fea_w = anchors_shape[1]
      num_anchors = anchors_shape[2] * fea_h * fea_w
      anchors = tf.reshape(anchors, [num_anchors, 4])  # reshape anchors
      anchors = xywh_to_yxyx(anchors)
      bbox = tf.constant([0.75, 0.75, 0.2, 0.2], dtype=tf.float32)
      bbox = xywh_to_yxyx(bbox)
      iou = batch_iou(anchors, bbox)
      anchor_idx = tf.arg_max(iou, dimension=0)
      anchors, output, anchor_idx = sess.run([anchors, iou, anchor_idx])
      print(anchors)
      print(output)
      print(anchor_idx)

  def test_batch_iou_(self):
    anchors = set_anchors(img_shape=[config.IMG_HEIGHT, config.IMG_WIDTH],
                          fea_shape=[config.FEA_HEIGHT, config.FEA_WIDTH])
    anchors_shape = anchors.get_shape().as_list()
    fea_h = anchors_shape[0]
    fea_w = anchors_shape[1]
    num_anchors = anchors_shape[2] * fea_h * fea_w
    anchors = tf.reshape(anchors, [num_anchors, 4])  # reshape anchors
    anchors = xywh_to_yxyx(anchors)
    bboxes = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    bboxes_ = xywh_to_yxyx(bboxes)
    ious, indices = batch_iou_(anchors, bboxes_)
    with self.test_session() as sess:
      ious, indices, bboxes_ = sess.run([ious, indices, bboxes], feed_dict={bboxes: [[0.25, 0.25, 0.5, 0.5],
                                                                                     [0.75, 0.75, 0.2, 0.2]]}
                                        )
      print(ious)
      print(indices)
      print(bboxes_)

  def test_batch_iou_fast(self):
    anchors = set_anchors(img_shape=[config.IMG_HEIGHT, config.IMG_WIDTH],
                          fea_shape=[config.FEA_HEIGHT, config.FEA_WIDTH])
    anchors_shape = anchors.get_shape().as_list()
    fea_h = anchors_shape[0]
    fea_w = anchors_shape[1]
    num_anchors = anchors_shape[2] * fea_h * fea_w
    anchors = tf.reshape(anchors, [num_anchors, 4])  # reshape anchors
    anchors = xywh_to_yxyx(anchors)
    bboxes = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    bboxes_ = xywh_to_yxyx(bboxes)
    ious, indices = batch_iou_fast(anchors, bboxes_)
    with self.test_session() as sess:
      ious, indices, bboxes_ = sess.run([ious, indices, bboxes],
                                        feed_dict={bboxes: [[0.07692308, 0.025, 0.13333334, 0.04025765],
                                                            [0.75, 0.75, 0.2, 0.2]]}
                                        )
      print(ious)
      print(indices)
      print(bboxes_)

  def test_encode_annos(self):
    with self.test_session() as sess:
      num_obj = 2
      image_shape = [config.IMG_HEIGHT, config.IMG_WIDTH]
      fea_shape = [config.FEA_HEIGHT, config.FEA_WIDTH]
      num_classes = config.NUM_CLASSES

      images = tf.constant(0, shape=[image_shape[0], image_shape[1], 3])
      labels = tf.constant(1, shape=[num_obj])
      anchors = set_anchors(image_shape, fea_shape)

      # Construct test bbox
      bbox_1 = tf.convert_to_tensor(xywh_to_yxyx(anchors[0][0][0]), dtype=tf.float32)
      bbox_2 = tf.convert_to_tensor(xywh_to_yxyx(anchors[2][2][1]), dtype=tf.float32)
      bboxes = tf.stack([bbox_1, bbox_2], axis=0)
      input_mask, labels_input, box_delta_input, box_input = \
        encode_annos(images, labels, bboxes, anchors, num_classes)
      out_input_mask, out_labels_input, out_box_delta_input, out_box_input = \
        sess.run([input_mask, labels_input, box_delta_input, box_input])
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
      anchors = set_anchors(img_shape=[config.IMG_HEIGHT, config.IMG_WIDTH],
                            fea_shape=[config.FEA_HEIGHT, config.FEA_WIDTH])
      output = sess.run(anchors)
      self.assertAllEqual(np.shape(output), [config.FEA_HEIGHT, config.FEA_WIDTH, config.NUM_ANCHORS, 4])
      print("Anchors:", output)
      print("Anchors shape:", np.shape(output))
      print("Num of anchors:", config.NUM_ANCHORS)