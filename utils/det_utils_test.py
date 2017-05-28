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
                                        feed_dict={bboxes: [[599.37, 212.45, 27.62, 25.34],
                                                            config.ANCHOR_SHAPE[2]]}
                                        )
      print("ious:", ious)
      print("max iou idx:", np.argmax(ious, axis=-1))
      print("bboxes:", bboxes_)

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
      bbox_2 = tf.convert_to_tensor(config.ANCHOR_SHAPE[-1], dtype=tf.float32)
      bboxes = tf.stack([bbox_1, bbox_2], axis=0)
      bboxes = xywh_to_yxyx(bboxes)
      anchors = xywh_to_yxyx(anchors)
      indices = arg_closest_anchor(bboxes, anchors)
      output = sess.run(indices)
      print('test_arg_closest_anchor')
      print(output)


  def test_update_tensor(self):
    with self.test_session() as sess:
      ref = tf.placeholder(dtype=tf.int64, shape=[None])#tf.convert_to_tensor([1, 2, 3], dtype=tf.int64)
      indices = tf.convert_to_tensor([0], dtype=tf.int64)
      update = tf.convert_to_tensor([9], dtype=tf.int64)
      tensor_updated = update_tensor(ref, indices, update)
      output = sess.run(tensor_updated, feed_dict={ref: [1, 2, 3]})
      print("test update tensor:")
      print("tensor updated", output)

  def test_encode_annos(self):
    with self.test_session() as sess:
      num_obj = 4
      num_classes = config.NUM_CLASSES

      labels = tf.constant(1, shape=[num_obj])
      anchors = tf.convert_to_tensor(config.ANCHOR_SHAPE, dtype=tf.float32)

      # Construct test bbox
      # bbox_1 = tf.convert_to_tensor(config.ANCHOR_SHAPE[0], dtype=tf.float32)
      # bbox_2 = tf.convert_to_tensor(config.ANCHOR_SHAPE[1], dtype=tf.float32)
      # bbox_3 = tf.convert_to_tensor(config.ANCHOR_SHAPE[2], dtype=tf.float32)
      # bbox_4 = tf.convert_to_tensor(config.ANCHOR_SHAPE[3], dtype=tf.float32)
      bbox_1 = tf.convert_to_tensor([599.37, 212.45, 27.62, 25.34], dtype=tf.float32)
      bbox_2 = tf.convert_to_tensor([922.14, 233.06, 91.41, 41.1], dtype=tf.float32)
      bbox_3 = tf.convert_to_tensor([862.41, 232.61, 86.96, 44.73], dtype=tf.float32)
      bbox_4 = tf.convert_to_tensor([729.87, 215.27, 32.22, 21.37], dtype=tf.float32)
      bboxes = tf.stack([bbox_1, bbox_2, bbox_3, bbox_4], axis=0)
      bboxes = xywh_to_yxyx(bboxes)

      input_mask, labels_input, box_delta_input, box_input = \
        encode_annos(labels, bboxes, anchors, num_classes)

      out_input_mask, out_labels_input, out_box_delta_input, out_box_input, out_anchors = \
        sess.run([input_mask, labels_input, box_delta_input, box_input, anchors])


      print(np.where(out_input_mask > 0))
      print(out_input_mask[2268], out_input_mask[2726], out_input_mask[2708], out_input_mask[2312])
      print(out_box_delta_input[2268], out_box_delta_input[2726], out_box_delta_input[2708], out_box_delta_input[2312])
      print(out_labels_input[2268], out_labels_input[2726], out_labels_input[2708], out_labels_input[2312])
      print(out_box_input[2268], out_box_input[2726], out_box_input[2708], out_box_input[2312])


