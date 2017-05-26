import tensorflow as tf
import numpy as np
from configs.kitti_config import config


# ################## From SqueezeDet #########################

def safe_exp(w, thresh):
  """Safe exponential function for tensors."""

  slope = np.exp(thresh)
  with tf.variable_scope('safe_exponential'):
    lin_region = tf.to_float(w > thresh)

    lin_out = slope * (w - thresh + 1.)
    exp_out = tf.exp(w)

    out = lin_region * lin_out + (1. - lin_region) * exp_out
  return out


def bbox_transform_inv(bbox):
  """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
  for numpy array or list of tensors.
  """
  with tf.variable_scope('bbox_transform_inv') as scope:
    xmin, ymin, xmax, ymax = bbox
    out_box = [[]] * 4

    width = xmax - xmin + 1.0
    height = ymax - ymin + 1.0
    out_box[0] = xmin + 0.5 * width
    out_box[1] = ymin + 0.5 * height
    out_box[2] = width
    out_box[3] = height

  return out_box


def bbox_transform(bbox):
  """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
  for numpy array or list of tensors.
  """
  with tf.variable_scope('bbox_transform') as scope:
    cx, cy, w, h = bbox
    out_box = [[]] * 4
    out_box[0] = cx - w / 2
    out_box[1] = cy - h / 2
    out_box[2] = cx + w / 2
    out_box[3] = cy + h / 2

  return out_box


# ################# MobileNet Det ########################

def xywh_to_yxyx(bbox):
  shape = bbox.get_shape().as_list()
  _axis = 1 if len(shape) > 1 else 0
  [x, y, w, h] = tf.unstack(bbox, axis=_axis)
  y_min = y - 0.5 * h
  x_min = x - 0.5 * w
  y_max = y + 0.5 * h
  x_max = x + 0.5 * w
  return tf.stack([y_min, x_min, y_max, x_max], axis=_axis)


def yxyx_to_xywh(bbox):
  y_min = bbox[:, 0]
  x_min = bbox[:, 1]
  y_max = bbox[:, 2]
  x_max = bbox[:, 3]
  x = (x_min + x_max) * 0.5
  y = (y_min + y_max) * 0.5
  w = x_max - x_min
  h = y_max - y_min
  return tf.stack([x, y, w, h], axis=1)


def rearrange_coords(bbox):
  y_min = bbox[:, 0]
  x_min = bbox[:, 1]
  y_max = bbox[:, 2]
  x_max = bbox[:, 3]
  return tf.stack([x_min, y_min, x_max, y_max])


def scale_bboxes(bbox, img_shape):
  """Scale bboxes to [0, 1). bbox format [ymin, xmin, ymax, xmax]
  Args:
    bbox: 2-D with shape '[num_bbox, 4]'
    img_shape: 1-D with shape '[4]'
  Return:
    sclaed_bboxes: scaled bboxes
  """
  img_h = tf.cast(img_shape[0], dtype=tf.float32)
  img_w = tf.cast(img_shape[1], dtype=tf.float32)
  shape = bbox.get_shape().as_list()
  _axis = 1 if len(shape) > 1 else 0
  [y_min, x_min, y_max, x_max] = tf.unstack(bbox, axis=_axis)
  y_1 = tf.truediv(y_min, img_h)
  x_1 = tf.truediv(x_min, img_w)
  y_2 = tf.truediv(y_max, img_h)
  x_2 = tf.truediv(x_max, img_w)
  return tf.stack([y_1, x_1, y_2, x_2], axis=_axis)


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


def batch_iou_(anchors, bboxes):
  """ Compute iou of two batch of boxes. Box format '[y_min, x_min, y_max, x_max]'.
  Args:
    anchors: know shape
    bboxes: dynamic shape
  Return:
    ious: 2-D with shape '[num_bboxes, num_anchors]'
    indices: [num_bboxes, 1]
  """
  num_anchors = anchors.get_shape().as_list()[0]
  ious_list = []
  for i in range(num_anchors):
    anchor = anchors[i]
    _ious = batch_iou(bboxes, anchor)
    ious_list.append(_ious)
  ious = tf.stack(ious_list, axis=0)
  ious = tf.transpose(ious)

  indices = tf.arg_max(ious, dimension=1)

  return ious, indices


def batch_iou_fast(anchors, bboxes):
  """ Compute iou of two batch of boxes. Box format '[y_min, x_min, y_max, x_max]'.
  Args:
    anchors: know shape
    bboxes: dynamic shape
  Return:
    ious: 2-D with shape '[num_bboxes, num_anchors]'
  """
  num_anchors = anchors.get_shape().as_list()[0]
  num_bboxes = tf.shape(bboxes)[0]

  box_indices = tf.reshape(tf.range(num_bboxes), shape=[-1, 1])
  box_indices = tf.reshape(tf.stack([box_indices] * num_anchors, axis=1), shape=[-1, 1])
  bboxes_m = tf.gather_nd(bboxes, box_indices)

  anchors_m = tf.tile(anchors, [num_bboxes, 1])

  print(bboxes_m.get_shape().as_list())
  lr = tf.maximum(
    tf.minimum(bboxes_m[:, 3], anchors_m[:, 3]) -
    tf.maximum(bboxes_m[:, 1], anchors_m[:, 1]),
    0
  )
  tb = tf.maximum(
    tf.minimum(bboxes_m[:, 2], anchors_m[:, 2]) -
    tf.maximum(bboxes_m[:, 0], anchors_m[:, 0]),
    0
  )

  intersection = tf.multiply(tb, lr)

  union = tf.subtract(
    tf.multiply((bboxes_m[:, 3] - bboxes_m[:, 1]), (bboxes_m[:, 2] - bboxes_m[:, 0])) +
    tf.multiply((anchors_m[:, 3] - anchors_m[:, 1]), (anchors_m[:, 2] - anchors_m[:, 0])),
    intersection
  )

  ious = tf.div(intersection, union)

  ious = tf.reshape(ious, shape=[num_bboxes, num_anchors])

  return ious


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


def batch_delta(bboxes, anchors):
  """
  Args:
     bboxes: [num_bboxes, 4]
     anchors: [num_bboxes, 4]
  Return:
    deltas: [num_bboxes, 4]
  """
  bbox_x, bbox_y, bbox_w, bbox_h = tf.unstack(bboxes, axis=1)
  anchor_x, anchor_y, anchor_w, anchor_h = tf.unstack(anchors, axis=1)
  delta_x = (bbox_x - anchor_x) / bbox_w
  delta_y = (bbox_y - anchor_y) / bbox_h
  delta_w = tf.log(bbox_w / anchor_w)
  delta_h = tf.log(bbox_h / anchor_h)
  return tf.stack([delta_x, delta_y, delta_w, delta_h], axis=1)


def arg_closest_anchor(bboxes, anchors):
  """Find the closest anchor. Box Format [ymin, xmin, ymax, xmax]
  """
  num_anchors = anchors.get_shape().as_list()[0]
  num_bboxes = tf.shape(bboxes)[0]

  _indices = tf.reshape(tf.range(num_bboxes), shape=[-1, 1])
  _indices = tf.reshape(tf.stack([_indices] * num_anchors, axis=1), shape=[-1, 1])
  bboxes_m = tf.gather_nd(bboxes, _indices)

  anchors_m = tf.tile(anchors, [num_bboxes, 1])

  square_dist = tf.squared_difference(bboxes_m[:, 0], anchors_m[:, 0]) + \
                tf.squared_difference(bboxes_m[:, 1], anchors_m[:, 1]) + \
                tf.squared_difference(bboxes_m[:, 2], anchors_m[:, 2]) + \
                tf.squared_difference(bboxes_m[:, 3], anchors_m[:, 3])

  square_dist = tf.reshape(square_dist, shape=[num_bboxes, num_anchors])

  indices = tf.arg_min(square_dist, dimension=1)

  return indices


def update_tensor(ref, indices, update):
  zero = tf.cast(tf.sparse_to_dense(indices,
                                    tf.shape(ref, out_type=tf.int64),
                                    0,
                                    default_value=1
                                    ),
                 dtype=tf.int64
                 )

  update_value = tf.cast(tf.sparse_to_dense(indices,
                                    tf.shape(ref, out_type=tf.int64),
                                    update,
                                    default_value=0
                                    ),
                 dtype=tf.int64
                 )
  return ref * zero + update_value


# TODO(shizehao): 1.deal with delta. 2.improve speed, use sparse tensor instead
def encode_annos(labels, bboxes, anchors, num_classes):
  """Encode annotations for losses computations.
  All the output tensors have a fix shape(none dynamic dimention).

  Args:
    labels: 1-D with shape `[num_bounding_boxes]`.
    bboxes: 2-D with shape `[num_bounding_boxes, 4]`. Format [ymin, xmin, ymax, xmax]
    anchors: 4-D tensor with shape `[fea_h, fea_w, num_anchors, 4]`. Format [cx, cy, w, h]

  Returns:
    input_mask: 2-D with shape `[num_anchors, 1]`, indicate which anchor to be used to cal loss.
    labels_input: 2-D with shape `[num_anchors, num_classes]`, one hot encode for every anchor.
    box_delta_input: 2-D with shape `[num_anchors, 4]`. Format [dcx, dcy, dw, dh]
    box_input: 2-D with shape '[num_anchors, 4]'. Format [ymin, xmin, ymax, xmax]
  """
  num_anchors = anchors.get_shape().as_list()[0]
  num_bboxes = tf.shape(bboxes)[0]

  # Cal iou, find the target anchor
  _anchors = xywh_to_yxyx(anchors)
  ious = batch_iou_fast(_anchors, bboxes)
  anchor_indices = tf.reshape(tf.arg_max(ious, dimension=1), shape=[-1, 1])  # target anchor indices

  # find the none-overlap bbox
  bbox_indices = tf.reshape(tf.range(num_bboxes), shape=[-1, 1])
  iou_indices = tf.concat([bbox_indices, tf.cast(anchor_indices, dtype=tf.int32)], axis=1)
  target_iou = tf.gather_nd(ious, iou_indices)
  none_overlap_bbox_indices = tf.where(target_iou <= 0) # 1-D


  # find it's corresponding anchor
  closest_anchor_indices = arg_closest_anchor(tf.gather_nd(bboxes, none_overlap_bbox_indices), anchors) # 1-D

  anchor_indices = tf.reshape(anchor_indices, shape=[-1])
  anchor_indices = update_tensor(anchor_indices, none_overlap_bbox_indices, closest_anchor_indices)
  anchor_indices = tf.reshape(anchor_indices, shape=[-1, 1])

  target_anchors = tf.gather_nd(anchors, anchor_indices)

  delta = batch_delta(yxyx_to_xywh(bboxes), target_anchors)

  # bbox
  box_input = tf.scatter_nd(
    anchor_indices,
    bboxes,
    shape=[num_anchors, 4]
  )

  # label
  labels_input = tf.scatter_nd(
    anchor_indices,
    tf.one_hot(labels, num_classes),
    shape=[num_anchors, num_classes]
  )

  # anchor mask
  onehot_anchor = tf.one_hot(anchor_indices, num_anchors)
  onehot_anchor = tf.squeeze(onehot_anchor, axis=1)
  input_mask = tf.reduce_sum(onehot_anchor, axis=0)
  input_mask = tf.reshape(input_mask, shape=[-1, 1])

  # delta
  box_delta_input = tf.scatter_nd(
    anchor_indices,
    delta,
    shape=[num_anchors, 4]
  )

  return input_mask, labels_input, box_delta_input, box_input
