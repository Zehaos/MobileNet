from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from configs import kitti_config as configs


def xywh_to_yxyx(bbox):
  [x, y, w, h] = tf.unstack(bbox)
  y_min = y - 0.5 * h
  x_min = x - 0.5 * w
  y_max = y + 0.5 * h
  x_max = x - 0.5 * w
  return tf.stack([y_min, x_min, y_max, x_max])


def yxyx_to_xywh(bbox):
  [y_min, x_min, y_max, x_max] = tf.unstack(bbox)
  x = 0.5 * (x_min + x_max)
  y = 0.5 * (y_min + y_max)
  w = x_max - x_min
  h = y_max - y_min
  return tf.stack([x, y, w, h])


def iou(bbox_1, bbox_2):
  """Compute iou of a box with another box. Box format '[y_min, x_min, y_max, x_max]'.
  Args:
    bbox_1: 1-D with shape `[4]`.
    bbox_2: 1-D with shape `[4]`.

  Returns:
    IOU
  """
  lr = tf.minimum(bbox_1[3], bbox_2[3]) - tf.maximum(bbox_1[1], bbox_2[1])
  if lr > 0:
    tb = tf.minimum(bbox_1[2], bbox_2[2]) - tf.maximum(bbox_1[0], bbox_2[0])
    if tb > 0:
      intersection = lr*tb
      union = (bbox_1[3]-bbox_1[1])*(bbox_1[2]-bbox_1[0])\
              +(bbox_2[3]-bbox_2[1])*(bbox_2[2]-bbox_2[0])\
              -intersection
      return intersection/union
  return 0



def batch_iou(bboxes, bbox):
  """Compute iou of a batch of boxes with another box. Box format '[y_min, x_min, y_max, x_max]'.
  Args:
    bboxes: A batch of boxes. 2-D with shape `[B, 4]`.
    bbox: A single box. 1-D with shape `[4]`.

  Returns:
    Batch of IOUs
  """
  lr = tf.maximum(
      tf.minimum(bboxes[:, 3], bbox[3]) - \
      tf.maximum(bboxes[:, 1], bbox[1]),
      0
  )
  tb = tf.maximum(
      tf.minimum(bboxes[:, 2], bbox[2]) - \
      tf.maximum(bboxes[:, 0], bbox[0]),
      0
  )
  intersection = lr*tb
  union = (bboxes[3] - bboxes[1]) * (bboxes[2] - bboxes[0]) \
          + (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) \
          - intersection
  return intersection/union


def encode_annos(image_shape, labels, bboxes, anchors):
  """Encode annotations for losses computations.

  Args:
    images: 4-D with shape `[B, H, W, C]`.
    labels: 2-D with shape `[B, num_bounding_boxes]`.
    bboxes: 3-D with shape `[B, num_bounding_boxes, 4]`.
    anchors: 4-D tensor with shape `[fea_h, fea_w, num_anchors, 4]`

  Returns:
    input_mask: 2-D with shape `[B, num_anchors]`, indicate which anchor to be used to cal loss.
    labels: 3-D with shape `[B, num_anchors, num_classes]`, one hot encode for every anchor.
    box_delta_input: 3-D with shape `[B, num_anchors, 4]`.
  """
  batch_size = image_shape[0]
  img_h = image_shape[1]
  img_w = image_shape[2]

  shape = anchors.get_shape().as_list()[0]
  fea_h = shape[0]
  fea_w = shape[1]
  num_anchors = shape[2]

  input_mask = tf.zeros([batch_size, fea_h, fea_w, num_anchors])
  labels = tf.zeros([batch_size, fea_h, fea_w, configs.NUM_CLASSES])
  box_delta_input = tf.zeros([batch_size, fea_h, fea_w, 4])

  # reshape to [batch, num_anchors

  return input_mask, labels, box_delta_input


def set_anchors(img_shape, fea_shape):
  """Set anchors.

  Args:
    img_shape: 1-D list with shape `[2]`.
    fea_shape: 1-D list with shape `[2]`.

  Returns:
    anchors: 4-D tensor with shape `[fea_h, fea_w, num_anchors, 4]`

  """
  img_h = img_shape[0]
  img_w = img_shape[1]
  fea_h = fea_shape[0]
  fea_w = fea_shape[1]

  anchor_shape = tf.constant(anchor_shape, dtype=tf.float32)
  anchor_shapes = tf.concat(
    [anchor_shape for i in range(fea_w * fea_h)], 0
  )
  anchor_shapes = tf.reshape(anchor_shapes, [fea_h, fea_w, 9, 2])

  center_x = tf.truediv(
    tf.range(1, fea_w + 1, 1, dtype=tf.float32) * img_w,
    float(fea_w + 1)
  )
  center_x = tf.concat(
    [center_x for i in range(fea_h * 9)], 0
  )
  center_x = tf.reshape(center_x, [fea_h, fea_w, 9, 1])

  center_y = tf.truediv(
    tf.range(1, fea_h + 1, 1, dtype=tf.float32) * img_h,
    float(fea_h + 1)
  )
  center_y = tf.concat(
    [center_y for i in range(fea_w * 9)], 0
  )
  center_y = tf.reshape(center_y, [fea_h, fea_w, 9, 1])

  anchors = tf.concat([center_x, center_y, anchor_shapes], 3)

  return anchors


def losses(input_mask, labels, ious, box_delta_input, pred_class_probs, pred_conf, pred_box_delta):
  num_objects = tf.reduce_sum(input_mask, name='num_objects')
  with tf.variable_scope('class_regression') as scope:
    # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
    # add a small value into log to prevent blowing up
    class_loss = tf.truediv(
      tf.reduce_sum(
        (labels * (-tf.log(pred_class_probs + configs.EPSILON))
         + (1 - labels) * (-tf.log(1 - pred_class_probs + configs.EPSILON)))
        * input_mask * configs.LOSS_COEF_CLASS),
      num_objects,
      name='class_loss'
    )
    tf.add_to_collection('losses', class_loss)

  with tf.variable_scope('confidence_score_regression') as scope:
    input_mask = tf.reshape(input_mask, [configs.BATCH_SIZE, configs.ANCHORS])
    conf_loss = tf.reduce_mean(
      tf.reduce_sum(
        tf.square((ious - pred_conf))
        * (input_mask * configs.LOSS_COEF_CONF_POS / num_objects
           + (1 - input_mask) * configs.LOSS_COEF_CONF_NEG / (configs.ANCHORS - num_objects)),
        reduction_indices=[1]
      ),
      name='confidence_loss'
    )
    tf.add_to_collection('losses', conf_loss)
    tf.summary.scalar('mean iou', tf.reduce_sum(ious) / num_objects)

  with tf.variable_scope('bounding_box_regression') as scope:
    bbox_loss = tf.truediv(
      tf.reduce_sum(
        configs.LOSS_COEF_BBOX * tf.square(
          input_mask * (pred_box_delta - box_delta_input))),
      num_objects,
      name='bbox_loss'
    )
    tf.add_to_collection('losses', bbox_loss)

  # add above losses as well as weight decay losses to form the total loss
  loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  return loss
