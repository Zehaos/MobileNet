from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
from utils.det_utils import *
from configs.kitti_config import config

import pickle

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
      bbox_1 = tf.convert_to_tensor([10, 10, 20, 20], dtype=tf.float32)
      bbox_2 = tf.convert_to_tensor([110, 110, 30, 30], dtype=tf.float32)
      bboxes = tf.stack([bbox_1, bbox_2], axis=0)

      anchor_1 = tf.convert_to_tensor([0,0,10,10], dtype=tf.float32)
      anchor_2 = tf.convert_to_tensor([100,100,110,110], dtype=tf.float32)
      anchors = tf.stack([anchor_1, anchor_2], axis=0)

      indices = arg_closest_anchor(bboxes, anchors)
      output = sess.run(indices)
      print('test_arg_closest_anchor')
      print(output)


  def test_update_tensor(self):
    with self.test_session() as sess:
      ref = tf.placeholder(dtype=tf.int64, shape=[None])#tf.convert_to_tensor([1, 2, 3], dtype=tf.int64)
      indices = tf.convert_to_tensor([2], dtype=tf.int64)
      update = tf.convert_to_tensor([9], dtype=tf.int64)
      tensor_updated = update_tensor(ref, indices, update)
      output = sess.run(tensor_updated, feed_dict={ref: [1, 2, 3]})
      print("test update tensor:")
      print("tensor updated", output)

  def test_encode_annos(self):
    with open("/home/zehao/PycharmProjects/MobileNet/utils/test_data.pkl", "rb") as fin:
      test_data = pickle.load(fin)

    with self.test_session() as sess:
      anchors = tf.convert_to_tensor(config.ANCHOR_SHAPE, dtype=tf.float32)

      num_image = len(test_data["test_bbox"])
      for i in range(50):
        bboxes = tf.convert_to_tensor(test_data["test_bbox"][i][0], dtype=tf.float32)
        bboxes = xywh_to_yxyx(bboxes)
        labels = tf.convert_to_tensor(test_data["test_label"][i][0])

        input_mask, labels_input, box_delta_input, box_input = \
          encode_annos(labels, bboxes, anchors, config.NUM_CLASSES)

        out_input_mask, out_labels_input, out_box_delta_input, out_box_input, out_anchors = \
          sess.run([input_mask, labels_input, box_delta_input, box_input, anchors])



        print("num_bbox:", np.shape(test_data["test_bbox"][i][0])[0])

        sd_indices = np.where(test_data["test_input_mask"][i][0] > 0)[1]
        print("SDet:")
        print("indices:", sd_indices)
        print("mask:", np.where(test_data["test_input_mask"][i][0] > 0)[1])
        print("bbox:", test_data["test_bbox"][i][0])
        print("label:", test_data["test_label"][i][0])
        print("delta:", test_data["test_input_delta"][i][0][0][sd_indices])
        print("first:", sd_indices[0], test_data["test_input_bbox"][i][0][0][sd_indices[0]], test_data["test_input_delta"][i][0][0][sd_indices[0]])

        indices = np.where(out_input_mask > 0)[0]
        print("Mine:")
        print("indices:", indices)
        print("mask:", np.where(out_input_mask > 0)[0])
        print("bbox:", out_box_input[indices])
        print("label:", out_labels_input[indices])
        print("delta:", out_box_delta_input[indices])
        print("first:", indices[0], out_box_input[indices[0]], out_box_delta_input[indices[0]])

        print("\n")
        # print("bbox:", out_box_input[indices])
        # aidx = np.where(test_data["test_input_mask"][i][0] > 0)[1]
        # encode_idx = np.where(out_input_mask > 0)[0]
        # flag = False
        # if np.shape(aidx)[0] != np.shape(encode_idx)[0]:
        #   flag = True
        # elif not np.alltrue(np.equal(aidx, encode_idx)):
        #   flag = True
        #   error_bidx = np.where(aidx != encode_idx)
        #   true_aidx = aidx[error_bidx]
        #   error_aidx = encode_idx[error_bidx]
        # if flag:
        #   image = test_data["test_image"][i][0]
        #   for b in range(np.shape(test_data["test_bbox"][i][0])[0]):
        #     bboxes = test_data["test_bbox"][i][0]
        #     bbox = bboxes[b]
        #     x = bbox[0]
        #     y = bbox[1]
        #     w = bbox[2]
        #     h = bbox[3]
        #     x1 = x-0.5*w
        #     y1 = y-0.5*h
        #     x2 = x+0.5*w
        #     y2 = y+0.5*h
        #     color = (255,0,0)
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        #     if np.any(error_bidx[0] == b):#b in error_bidx:
        #       for a in config.ANCHOR_SHAPE[true_aidx]:
        #         true_a = a
        #         x = true_a[0]
        #         y = true_a[1]
        #         w = true_a[2]
        #         h = true_a[3]
        #         x1 = x - 0.5 * w
        #         y1 = y - 0.5 * h
        #         x2 = x + 0.5 * w
        #         y2 = y + 0.5 * h
        #         color = (0, 255, 255)
        #         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        #       for ea in config.ANCHOR_SHAPE[error_aidx]:
        #         error_a = ea
        #         x = error_a[0]
        #         y = error_a[1]
        #         w = error_a[2]
        #         h = error_a[3]
        #         x1 = x - 0.5 * w
        #         y1 = y - 0.5 * h
        #         x2 = x + 0.5 * w
        #         y2 = y + 0.5 * h
        #         color = (255, 255, 0)
        #         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        #   # cv2.imwrite("img" + str(b) + ".jpg", image)
        #   cv2.imshow("img", image)
        #   cv2.waitKey(0)

  def test_set_anchors(self):
    anchors = config.ANCHOR_SHAPE
    image = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, 3))
    num_anchors = np.shape(anchors)[0]
    for i in range(num_anchors):
      anchor = anchors[i]
      x = anchor[0]
      y = anchor[1]
      w = anchor[2]
      h = anchor[3]
      x1 = x - 0.5*w
      y1 = y - 0.5*h
      x2 = x + 0.5*w
      y2 = y + 0.5*h
      cv2.rectangle(image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255,255,255)
                    )
      cv2.rectangle(image,
                    (int(739.72003), int(181.11)),
                    (int(770.04), int(204.92)),
                    (255, 0, 0),
                    2)
      if i == 2313:
        cv2.rectangle(image,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 255),
                      2
                      )
    cv2.imshow("anchors", image)
    cv2.waitKey(0)