from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils.det_utils import *
from configs.kitti_config import config


class MobileNetDetTest(tf.test.TestCase):
  def test_batch_iou_fast(self):
    anchors = tf.convert_to_tensor(config.ANCHOR_SHAPE, dtype=tf.float32)
    anchors = xywh_to_yxyx(anchors)
    bboxes = tf.placeholder(dtype=tf.float32, shape=[None, 4])
    bboxes_ = xywh_to_yxyx(bboxes)
    ious = batch_iou_fast(anchors, bboxes_)
    with self.test_session() as sess:
      ious, bboxes_ = sess.run([ious, bboxes],
                                        feed_dict={bboxes: [config.ANCHOR_SHAPE[0],
                                                            config.ANCHOR_SHAPE[-1]]}
                                        )
      print(ious)
      print(bboxes_)

  def test_set_anchors(self):
    with self.test_session() as sess:
      anchors = tf.convert_to_tensor(config.ANCHOR_SHAPE, dtype=tf.float32)
      output = sess.run(anchors)
      print(np.shape(output))
      self.assertAllEqual(np.shape(output), [config.FEA_HEIGHT * config.FEA_WIDTH * config.NUM_ANCHORS, 4])
      print("Anchors:", output)
      print("Anchors shape:", np.shape(output))
      print("Num of anchors:", config.NUM_ANCHORS)

  def test_arg_closest_anchor(self):
    with self.test_session() as sess:
      anchors = tf.convert_to_tensor(config.ANCHOR_SHAPE, dtype=tf.float32)
      bbox_1 = tf.convert_to_tensor(config.ANCHOR_SHAPE[0], dtype=tf.float32)
      bbox_2 = tf.convert_to_tensor([646.1, 2210.9, 73., 44.], dtype=tf.float32)
      bboxes = tf.stack([bbox_1, bbox_2], axis=0)
      bboxes = xywh_to_yxyx(bboxes)
      indices = arg_closest_anchor(bboxes, anchors)
      output = sess.run(indices)
      print('test_arg_closest_anchor')
      print(output)


  def test_update_tensor(self):
    with self.test_session() as sess:
      ref = tf.convert_to_tensor([[1], [2], [3]], dtype=tf.int64)
      indices = tf.convert_to_tensor([[0]], dtype=tf.int64)
      update = tf.convert_to_tensor([[9]], dtype=tf.int64)
      tensor_updated = update_tensor(ref, indices, update)
      output = sess.run(tensor_updated)
      print("test update tensor:")
      print("tensor updated", output)

  def test_encode_annos(self):
    with self.test_session() as sess:
      num_obj = 2
      num_classes = config.NUM_CLASSES

      labels = tf.constant(1, shape=[num_obj])
      anchors = tf.convert_to_tensor(config.ANCHOR_SHAPE, dtype=tf.float32)

      # Construct test bbox
      bbox_1 = tf.convert_to_tensor(config.ANCHOR_SHAPE[0], dtype=tf.float32)
      bbox_2 = tf.convert_to_tensor([646.1, 2210.9, 73., 44.], dtype=tf.float32)
      bboxes = tf.stack([bbox_1, bbox_2], axis=0)
      bboxes = xywh_to_yxyx(bboxes)

      input_mask, labels_input, box_delta_input, box_input = \
        encode_annos(labels, bboxes, anchors, num_classes)

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

