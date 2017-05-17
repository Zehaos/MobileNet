from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from configs.kitti_config import config


def xywh_to_yxyx(bbox):
  [x, y, w, h] = tf.unstack(bbox)
  y_min = y - 0.5 * h
  x_min = x - 0.5 * w
  y_max = y + 0.5 * h
  x_max = x + 0.5 * w
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
  tb = tf.minimum(bbox_1[2], bbox_2[2]) - tf.maximum(bbox_1[0], bbox_2[0])
  lr = tf.maximum(lr, lr * 0)
  tb = tf.maximum(tb, tb * 0)
  intersection = tf.multiply(tb, lr)
  union = tf.subtract(
    tf.multiply((bbox_1[3] - bbox_1[1]), (bbox_1[2] - bbox_1[0])) +
    tf.multiply((bbox_2[3] - bbox_2[1]), (bbox_2[2] - bbox_2[0])),
    intersection
  )
  iou = tf.div(intersection, union)
  return iou


def batch_iou(bboxes, bbox):
  """Compute iou of a batch of boxes with another box. Box format '[y_min, x_min, y_max, x_max]'.
  Args:
    bboxes: A batch of boxes. 2-D with shape `[B, 4]`.
    bbox: A single box. 1-D with shape `[4]`.

  Returns:
    Batch of IOUs
  """
  num_bboxes = bboxes.get_shape().as_list()[0]
  lr = tf.maximum(
    tf.minimum(bboxes[:, 3], bbox[3]) -
    tf.maximum(bboxes[:, 1], bbox[1]),
    0
  )
  tb = tf.maximum(
    tf.minimum(bboxes[:, 2], bbox[2]) -
    tf.maximum(bboxes[:, 0], bbox[0]),
    0
  )
  intersection = tf.multiply(tb, lr)
  union = tf.subtract(
    tf.multiply((bboxes[:, 3] - bboxes[:, 1]), (bboxes[:, 2] - bboxes[:, 0])) +
    tf.multiply((bbox[3] - bbox[1]), (bbox[2] - bbox[0])),
    intersection
  )
  iou = tf.div(intersection, union)
  return iou


def compute_delta(gt_box, anchor):
  """Compute delta, anchor+delta = gt_box. Box format '[cx, cy, w, h]'.
  Args:
    gt_box: A batch of boxes. 2-D with shape `[B, 4]`.
    anchor: A single box. 1-D with shape `[4]`.

  Returns:
    delta: 1-D tensor with shape '[4]', [dx, dy, dw, dh]
  """
  delta_x = (gt_box[0] - anchor[0]) / gt_box[2]
  delta_y = (gt_box[1] - anchor[1]) / gt_box[3]
  delta_w = tf.log(gt_box[2] / anchor[2])
  delta_h = tf.log(gt_box[3] / anchor[3])
  return tf.stack([delta_x, delta_y, delta_w, delta_h], axis=0)


def encode_annos(images, labels, bboxes, anchors, num_classes):
  """Encode annotations for losses computations.

  Args:
    images: 4-D with shape `[B, H, W, C]`.
    labels: 2-D with shape `[B, num_bounding_boxes]`.
    bboxes: 3-D with shape `[B, num_bounding_boxes, 4]`.
    anchors: 4-D tensor with shape `[fea_h, fea_w, num_anchors, 4]`

  Returns:
    input_mask: 2-D with shape `[B, num_anchors]`, indicate which anchor to be used to cal loss.
    labels_input: 3-D with shape `[B, num_anchors, num_classes]`, one hot encode for every anchor.
    box_delta_input: 3-D with shape `[B, num_anchors, 4]`.
    box_input: 3-D with shape '[B, num_anchors, 4]'.
  """
  images_shape = images.get_shape().as_list()[0]
  batch_size = images_shape[0]
  img_h = images_shape[1]
  img_w = images_shape[2]

  anchors_shape = anchors.get_shape().as_list()[0]
  fea_h = anchors_shape[0]
  fea_w = anchors_shape[1]
  num_anchors = anchors_shape[2]

  bboxes_shape = bboxes.get_shape().as_list()[0]
  num_obj = bboxes_shape[1]

  anchor_idx_list = []
  onehot_aid_list = []
  onehot_labels_list = []
  batch_onehot_labels_list = []
  batch_bboxes_list = []
  bbox_list = []
  delta_list = []
  batch_delta_list = []
  for i in range(batch_size):  # for each image
    for j in range(num_obj):  # for each bbox
      # bbox
      bbox = bboxes[i][j]
      bbox_list.append(bbox)
      # label
      label = labels[i][j]
      onehot_labels_list.append(tf.one_hot(label, num_classes))  # collect anchor cls
      # anchor
      ious = batch_iou(anchors, bbox)  # TODO(shizehao): reshape anchors
      anchors_idx = tf.arg_max(ious, dimension=0)  # find the target anchor
      anchor_idx_list.append(anchors_idx)  # collect anchor idx
      # delta
      delta_list.append(compute_delta(bbox, anchors[anchors_idx]))

    # bbox
    batch_bboxes_list.append(
      tf.expand_dims(
        tf.scatter_nd(tf.constant(anchor_idx_list), bbox_list),
        0)
    )
    # label
    batch_onehot_labels_list.append(
      tf.expand_dims(
        tf.scatter_nd(tf.constant(anchor_idx_list), onehot_labels_list),
        0)
    )
    # anchor
    onehot_anchor = tf.one_hot(anchor_idx_list, num_anchors)
    onehot_aid_list.append(tf.reduce_sum(onehot_anchor, axis=0))
    # delta
    batch_delta_list.append(
      tf.expand_dims(
        tf.scatter_nd(tf.constant(anchor_idx_list), delta_list),
        0)
    )

  input_mask = tf.stack(onehot_aid_list, axis=0)
  labels_input = tf.stack(batch_onehot_labels_list, axis=0)
  box_input = tf.stack(batch_bboxes_list, axis=0)
  box_delta_input = tf.stack(batch_delta_list, axis=0)

  return input_mask, labels_input, box_delta_input, box_input


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

  anchor_shape = tf.constant(config.ANCHOR_SHAPE, dtype=tf.float32)
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
