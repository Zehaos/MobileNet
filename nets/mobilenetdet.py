from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from configs.kitti_config import config


def xywh_to_yxyx(bbox):
  shape = bbox.get_shape().as_list()
  _axis = 1 if len(shape) > 1 else 0
  [x, y, w, h] = tf.unstack(bbox, axis=_axis)
  y_min = y - 0.5 * h
  x_min = x - 0.5 * w
  y_max = y + 0.5 * h
  x_max = x + 0.5 * w
  return tf.stack([y_min, x_min, y_max, x_max], axis=_axis)


def yxyx_to_xywh_(bbox):
  y_min = bbox[:, 0]
  x_min = bbox[:, 1]
  y_max = bbox[:, 2]
  x_max = bbox[:, 3]
  x = (x_min + x_max) * 0.5
  y = (y_min + y_max) * 0.5
  w = x_max - x_min
  h = y_max - y_min
  return tf.stack([x, y, w, h], axis=1)


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
    indices: [num_bboxes, 1]
  """
  num_anchors = anchors.get_shape().as_list()[0]
  tensor_num_bboxes = tf.shape(bboxes)[0]
  indices = tf.reshape(tf.range(tensor_num_bboxes), shape=[-1, 1])
  indices = tf.reshape(tf.stack([indices]*num_anchors, axis=1), shape=[-1, 1])
  bboxes_m = tf.gather_nd(bboxes, indices)

  anchors_m = tf.tile(anchors, [tensor_num_bboxes, 1])

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

  ious = tf.reshape(ious, shape=[tensor_num_bboxes, num_anchors])

  indices = tf.arg_max(ious, dimension=1)

  return ious, indices


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


# TODO(shizehao): improve speed, use sparse tensor instead
def encode_annos(image, labels, bboxes, anchors, num_classes):
  """Encode annotations for losses computations.
  All the output tensors have a fix shape(none dynamic dimention).

  Args:
    image: 4-D with shape `[H, W, C]`.
    b_labels: 2-D with shape `[num_bounding_boxes]`.
    b_bboxes: 3-D with shape `[num_bounding_boxes, 4]`. Scaled.
    anchors: 4-D tensor with shape `[fea_h, fea_w, num_anchors, 4]`

  Returns:
    input_mask: 2-D with shape `[num_anchors, 1]`, indicate which anchor to be used to cal loss.
    labels_input: 2-D with shape `[num_anchors, num_classes]`, one hot encode for every anchor.
    box_delta_input: 2-D with shape `[num_anchors, 4]`.
    box_input: 2-D with shape '[num_anchors, 4]'.
  """
  anchors_shape = anchors.get_shape().as_list()
  fea_h = anchors_shape[0]
  fea_w = anchors_shape[1]
  num_anchors = anchors_shape[2] * fea_h * fea_w
  anchors = tf.reshape(anchors, [num_anchors, 4])  # reshape anchors

  # Cal iou, find the target anchor
  _anchors = xywh_to_yxyx(anchors)
  ious, indices = batch_iou_fast(_anchors, bboxes)
  indices = tf.reshape(indices, shape=[-1, 1])

  target_anchors = tf.gather(anchors, indices)
  target_anchors = tf.squeeze(target_anchors, axis=1)
  delta = batch_delta(yxyx_to_xywh_(bboxes), target_anchors)

  # bbox
  box_input = tf.scatter_nd(
    indices,
    bboxes,
    shape=[num_anchors, 4]
  )
  # label
  labels_input = tf.scatter_nd(
    indices,
    tf.one_hot(labels, num_classes),
    shape=[num_anchors, num_classes]
  )
  # anchor mask
  onehot_anchor = tf.one_hot(indices, num_anchors)
  onehot_anchor = tf.squeeze(onehot_anchor, axis=1)
  print("indices shape:", indices.get_shape().as_list())
  print("one hot anchors shape:", onehot_anchor.get_shape().as_list())
  input_mask = tf.reduce_sum(onehot_anchor, axis=0)
  input_mask = tf.reshape(input_mask, shape=[-1, 1])
  # delta
  box_delta_input = tf.scatter_nd(
    indices,
    delta,
    shape=[num_anchors, 4]
  )

  return input_mask, labels_input, box_delta_input, box_input


# TODO(shizehao): align anchor center to the grid
def set_anchors(img_shape, fea_shape):
  """Set anchors.

  Args:
    img_shape: 1-D list with shape `[2]`.
    fea_shape: 1-D list with shape `[2]`.

  Returns:
    anchors: 4-D tensor with shape `[fea_h, fea_w, num_anchors, 4]`

  """
  H = fea_shape[0]
  W = fea_shape[1]
  B = config.NUM_ANCHORS

  anchor_shape = tf.constant(config.ANCHOR_SHAPE, dtype=tf.float32)
  anchor_shapes = tf.reshape(
    tf.concat(
      [anchor_shape for i in range(W * H)],
      0
    ),
    [H, W, B, 2]
  )

  center_x = tf.truediv(
    tf.range(1, W + 1, 1, dtype=tf.float32),  # * img_w,
    float(W + 1)
  )
  center_x = tf.concat(
    [center_x for i in range(H * B)], 0
  )
  center_x = tf.reshape(center_x, [B, H, W])
  center_x = tf.transpose(center_x, (1, 2, 0))
  center_x = tf.reshape(center_x, [H, W, B, 1])

  center_y = tf.truediv(
    tf.range(1, H + 1, 1, dtype=tf.float32),  # * img_h,
    float(H + 1)
  )
  center_y = tf.concat(
    [center_y for i in range(W * B)], 0
  )
  center_y = tf.reshape(center_y, [B, W, H])
  center_y = tf.transpose(center_y, (2, 1, 0))
  center_y = tf.reshape(center_y, [H, W, B, 1])

  anchors = tf.concat([center_x, center_y, anchor_shapes], 3)

  return anchors


def interpre_prediction(prediction, input_mask, anchors, box_input, fea_h, fea_w):
  # probability
  batch_size = tf.shape(input_mask)[0]
  num_class_probs = config.NUM_ANCHORS * config.NUM_CLASSES
  pred_class_probs = tf.reshape(
    tf.nn.softmax(
      tf.reshape(
        prediction[:, :, :, :num_class_probs],
        [-1, config.NUM_CLASSES]
      )
    ),
    [batch_size, config.NUM_ANCHORS * fea_h * fea_w, config.NUM_CLASSES],
    name='pred_class_probs'
  )

  # confidence
  num_confidence_scores = config.NUM_ANCHORS + num_class_probs
  pred_conf = tf.sigmoid(
    tf.reshape(
      prediction[:, :, :, num_class_probs:num_confidence_scores],
      [batch_size, config.NUM_ANCHORS * fea_h * fea_w]
    ),
    name='pred_confidence_score'
  )

  # bbox_delta
  pred_box_delta = tf.reshape(
    prediction[:, :, :, num_confidence_scores:],
    [batch_size, config.NUM_ANCHORS * fea_h * fea_w, 4],
    name='bbox_delta'
  )

  # number of object. Used to normalize bbox and classification loss
  num_objects = tf.reduce_sum(input_mask, name='num_objects')

  with tf.variable_scope('bbox') as scope:
    with tf.variable_scope('stretching'):
      delta_x, delta_y, delta_w, delta_h = tf.unstack(
        pred_box_delta, axis=2)

      anchor_x = anchors[:, 0]
      anchor_y = anchors[:, 1]
      anchor_w = anchors[:, 2]
      anchor_h = anchors[:, 3]

      box_center_x = tf.identity(
        anchor_x + delta_x * anchor_w, name='bbox_cx')
      box_center_y = tf.identity(
        anchor_y + delta_y * anchor_h, name='bbox_cy')
      box_width = tf.identity(
        anchor_w * safe_exp(delta_w, config.EXP_THRESH),
        name='bbox_width')
      box_height = tf.identity(
        anchor_h * safe_exp(delta_h, config.EXP_THRESH),
        name='bbox_height')

    with tf.variable_scope('trimming'):
      xmins, ymins, xmaxs, ymaxs = bbox_transform(
        [box_center_x, box_center_y, box_width, box_height])

      # The max x position is mc.IMAGE_WIDTH - 1 since we use zero-based
      # pixels. Same for y.
      xmins = tf.minimum(
        tf.maximum(0.0, xmins), config.IMG_WIDTH - 1.0, name='bbox_xmin')

      ymins = tf.minimum(
        tf.maximum(0.0, ymins), config.IMG_HEIGHT - 1.0, name='bbox_ymin')

      xmaxs = tf.maximum(
        tf.minimum(config.IMG_WIDTH - 1.0, xmaxs), 0.0, name='bbox_xmax')

      ymaxs = tf.maximum(
        tf.minimum(config.IMG_HEIGHT - 1.0, ymaxs), 0.0, name='bbox_ymax')

      det_boxes = tf.transpose(
        tf.stack(bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
        (1, 2, 0), name='bbox'
      )

      with tf.variable_scope('IOU'):
        def _tensor_iou(box1, box2):
          with tf.variable_scope('intersection'):
            xmin = tf.maximum(box1[0], box2[0], name='xmin')
            ymin = tf.maximum(box1[1], box2[1], name='ymin')
            xmax = tf.minimum(box1[2], box2[2], name='xmax')
            ymax = tf.minimum(box1[3], box2[3], name='ymax')

            w = tf.maximum(0.0, xmax - xmin, name='inter_w')
            h = tf.maximum(0.0, ymax - ymin, name='inter_h')
            intersection = tf.multiply(w, h, name='intersection')

          with tf.variable_scope('union'):
            w1 = tf.subtract(box1[2], box1[0], name='w1')
            h1 = tf.subtract(box1[3], box1[1], name='h1')
            w2 = tf.subtract(box2[2], box2[0], name='w2')
            h2 = tf.subtract(box2[3], box2[1], name='h2')

            union = w1 * h1 + w2 * h2 - intersection

          return intersection / (union + config.EPSILON) \
                 * tf.reshape(input_mask, [batch_size, config.NUM_ANCHORS * fea_h * fea_w])

        # TODO(shizehao): need test
        ious = _tensor_iou(
          bbox_transform(tf.unstack(det_boxes, axis=2)),
          bbox_transform(tf.unstack(box_input, axis=2))
        )

      with tf.variable_scope('probability') as scope:
        probs = tf.multiply(
          pred_class_probs,
          tf.reshape(pred_conf, [batch_size, config.NUM_ANCHORS * fea_h * fea_w, 1]),
          name='final_class_prob'
        )

        det_probs = tf.reduce_max(probs, 2, name='score')
        det_class = tf.argmax(probs, 2, name='class_idx')

  return pred_box_delta, pred_class_probs, pred_conf, ious, det_probs, det_boxes, det_class


def losses(input_mask, labels, ious, box_delta_input, pred_class_probs, pred_conf, pred_box_delta):
  batch_size = tf.shape(input_mask)[0]
  num_objects = tf.reduce_sum(input_mask, name='num_objects')
  with tf.variable_scope('class_regression') as scope:
    # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
    # add a small value into log to prevent blowing up
    class_loss = tf.truediv(
      tf.reduce_sum(
        (labels * (-tf.log(pred_class_probs + config.EPSILON))
         + (1 - labels) * (-tf.log(1 - pred_class_probs + config.EPSILON)))
        * input_mask * config.LOSS_COEF_CLASS),
      num_objects,
      name='class_loss'
    )
    tf.losses.add_loss(class_loss)

  with tf.variable_scope('confidence_score_regression') as scope:
    input_mask_ = tf.reshape(input_mask, [batch_size, config.ANCHORS])
    conf_loss = tf.reduce_mean(
      tf.reduce_sum(
        tf.square((ious - pred_conf))
        * (input_mask_ * config.LOSS_COEF_CONF_POS / num_objects
           + (1 - input_mask_) * config.LOSS_COEF_CONF_NEG / (config.ANCHORS - num_objects)),
        reduction_indices=[1]
      ),
      name='confidence_loss'
    )
    tf.losses.add_loss(conf_loss)

  with tf.variable_scope('bounding_box_regression') as scope:
    bbox_loss = tf.truediv(
      tf.reduce_sum(
        config.LOSS_COEF_BBOX * tf.square(
          input_mask * (pred_box_delta - box_delta_input))),
      num_objects,
      name='bbox_loss'
    )
    tf.losses.add_loss(bbox_loss)

  # add above losses as well as weight decay losses to form the total loss
  loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  return loss


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


def mobilenet(inputs,
          is_training=True,
          width_multiplier=1,
          scope='MobileNet'):
  def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                width_multiplier,
                                sc,
                                downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn

  with tf.variable_scope(scope) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None,
                        outputs_collections=[end_points_collection]):
      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          activation_fn=tf.nn.relu):
        net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
        net = slim.batch_norm(net, scope='conv_1/batch_norm')
        net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
        net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
        net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
        net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
        net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
        net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

        net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
        net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

  return end_points


def mobilenet_arg_scope(weight_decay=0.0):
  """Defines the default mobilenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the MobileNet model.
  """
  with slim.arg_scope(
      [slim.convolution2d, slim.separable_convolution2d],
      weights_initializer=slim.initializers.xavier_initializer(),
      biases_initializer=slim.init_ops.zeros_initializer(),
      weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
    return sc