import tensorflow as tf
import numpy as np
import cv2
from configs.kitti_config import config


# ################## From SqueezeDet #########################

def safe_exp(w, thresh):
  """Safe exponential function for tensors."""

  slope = np.exp(thresh)
  with tf.name_scope('safe_exponential'):
    lin_region = tf.to_float(w > thresh)

    lin_out = slope * (w - thresh + 1.)
    exp_out = tf.exp(w)

    out = lin_region * lin_out + (1. - lin_region) * exp_out
  return out


def bbox_transform_inv(bbox):
  """convert a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]. Works
  for numpy array or list of tensors.
  """
  with tf.name_scope('bbox_transform_inv') as scope:
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
  with tf.name_scope('bbox_transform') as scope:
    cx, cy, w, h = bbox
    out_box = [[]] * 4
    out_box[0] = cx - w / 2
    out_box[1] = cy - h / 2
    out_box[2] = cx + w / 2
    out_box[3] = cy + h / 2

  return out_box


def nms(boxes, probs, threshold):
  """Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """

  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    for j, ov in enumerate(ovps):
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep


def interpre_prediction(prediction, input_mask, anchors, box_input):
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
    [batch_size, config.ANCHORS, config.NUM_CLASSES],
    name='pred_class_probs'
  )

  # confidence
  num_confidence_scores = config.NUM_ANCHORS + num_class_probs
  pred_conf = tf.sigmoid(
    tf.reshape(
      prediction[:, :, :, num_class_probs:num_confidence_scores],
      [batch_size, config.ANCHORS]
    ),
    name='pred_confidence_score'
  )

  # bbox_delta
  pred_box_delta = tf.reshape(
    prediction[:, :, :, num_confidence_scores:],
    [batch_size, config.ANCHORS, 4],
    name='bbox_delta'
  )

  # number of object. Used to normalize bbox and classification loss
  num_objects = tf.reduce_sum(input_mask, name='num_objects')

  with tf.name_scope('bbox') as scope:
    with tf.name_scope('stretching'):
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

    with tf.name_scope('trimming'):
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

      with tf.name_scope('IOU'):
        def _tensor_iou(box1, box2):
          with tf.name_scope('intersection'):
            xmin = tf.maximum(box1[0], box2[0], name='xmin')
            ymin = tf.maximum(box1[1], box2[1], name='ymin')
            xmax = tf.minimum(box1[2], box2[2], name='xmax')
            ymax = tf.minimum(box1[3], box2[3], name='ymax')

            w = tf.maximum(0.0, xmax - xmin, name='inter_w')
            h = tf.maximum(0.0, ymax - ymin, name='inter_h')
            intersection = tf.multiply(w, h, name='intersection')

          with tf.name_scope('union'):
            w1 = tf.subtract(box1[2], box1[0], name='w1')
            h1 = tf.subtract(box1[3], box1[1], name='h1')
            w2 = tf.subtract(box2[2], box2[0], name='w2')
            h2 = tf.subtract(box2[3], box2[1], name='h2')

            union = w1 * h1 + w2 * h2 - intersection

          return intersection / (union + config.EPSILON) \
                 * tf.reshape(input_mask, [batch_size, config.ANCHORS])

        ious = _tensor_iou(
          bbox_transform(tf.unstack(det_boxes, axis=2)),
          bbox_transform(tf.unstack(box_input, axis=2))
        )

      with tf.name_scope('probability') as scope:
        probs = tf.multiply(
          pred_class_probs,
          tf.reshape(pred_conf, [batch_size, config.ANCHORS, 1]),
          name='final_class_prob'
        )

        det_probs = tf.reduce_max(probs, 2, name='score')
        det_class = tf.argmax(probs, 2, name='class_idx')

  return pred_box_delta, pred_class_probs, pred_conf, ious, det_probs, det_boxes, det_class


def losses(input_mask, labels, ious, box_delta_input, pred_class_probs, pred_conf, pred_box_delta):
  batch_size = tf.shape(input_mask)[0]
  num_objects = tf.reduce_sum(input_mask, name='num_objects')

  with tf.name_scope('class_regression') as scope:
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

  with tf.name_scope('confidence_score_regression') as scope:
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

  with tf.name_scope('bounding_box_regression') as scope:
    bbox_loss = tf.truediv(
      tf.reduce_sum(
        config.LOSS_COEF_BBOX * tf.square(
          input_mask * (pred_box_delta - box_delta_input))),
      num_objects,
      name='bbox_loss'
    )
    tf.losses.add_loss(bbox_loss)

    # # add above losses as well as weight decay losses to form the total loss
    # loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    # return loss


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
  box_indices = tf.reshape(tf.stack([box_indices] * num_anchors, axis=1), shape=[-1, 1]) # use tf.tile instead
  # box_indices = tf.tile(box_indices, [num_anchors, 1])

  # box_indices = tf.Print(box_indices, [box_indices], "box_indices", summarize=100)


  bboxes_m = tf.gather_nd(bboxes, box_indices)
  # bboxes_m = tf.Print(bboxes_m, [bboxes_m], "bboxes_m", summarize=100)

  anchors_m = tf.tile(anchors, [num_bboxes, 1])
  # anchors_m = tf.Print(anchors_m, [anchors_m], "anchors_m", summarize=100)

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

  ious = tf.truediv(intersection, union)

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
  # bboxes_m = tf.Print(bboxes_m, [bboxes_m], "bboxes_m", summarize=100)

  anchors_m = tf.tile(anchors, [num_bboxes, 1])
  # anchors_m = tf.Print(anchors_m, [anchors_m], "anchors_m", summarize=100)

  square_dist = tf.squared_difference(bboxes_m[:, 0], anchors_m[:, 0]) + \
                tf.squared_difference(bboxes_m[:, 1], anchors_m[:, 1]) + \
                tf.squared_difference(bboxes_m[:, 2], anchors_m[:, 2]) + \
                tf.squared_difference(bboxes_m[:, 3], anchors_m[:, 3])

  square_dist = tf.reshape(square_dist, shape=[num_bboxes, num_anchors])

  # square_dist = tf.Print(square_dist, [square_dist], "square_dist", summarize=100)

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


def find_dup(a):
  """ Find the duplicated elements in 1-D a tensor.
  Args:
    a: 1-D tensor.
    
  Return:
    more_than_one_vals: duplicated value in a.
    indexes_in_a: duplicated value's index in a.
    dups_in_a: duplicated value with duplicate in a.
  """
  unique_a_vals, unique_idx = tf.unique(a)
  count_a_unique = tf.unsorted_segment_sum(tf.ones_like(a),
                                           unique_idx,
                                           tf.shape(a)[0])

  more_than_one = tf.greater(count_a_unique, 1)
  more_than_one_idx = tf.squeeze(tf.where(more_than_one))
  more_than_one_vals = tf.squeeze(tf.gather(unique_a_vals, more_than_one_idx))

  not_duplicated, _ = tf.setdiff1d(a, more_than_one_vals)
  dups_in_a, indexes_in_a = tf.setdiff1d(a, not_duplicated)

  return more_than_one_vals, indexes_in_a, dups_in_a


def encode_annos(labels, bboxes, anchors, num_classes):
  """Encode annotations for losses computations.
  All the output tensors have a fix shape(none dynamic dimention).

  Args:
    labels: 1-D with shape `[num_bounding_boxes]`.
    bboxes: 2-D with shape `[num_bounding_boxes, 4]`. Format [ymin, xmin, ymax, xmax]
    anchors: 4-D tensor with shape `[num_anchors, 4]`. Format [cx, cy, w, h]

  Returns:
    input_mask: 2-D with shape `[num_anchors, 1]`, indicate which anchor to be used to cal loss.
    labels_input: 2-D with shape `[num_anchors, num_classes]`, one hot encode for every anchor.
    box_delta_input: 2-D with shape `[num_anchors, 4]`. Format [dcx, dcy, dw, dh]
    box_input: 2-D with shape '[num_anchors, 4]'. Format [ymin, xmin, ymax, xmax]
  """
  with tf.name_scope("Encode_annotations") as scope:
    num_anchors = config.ANCHORS
    # num_bboxes = tf.shape(bboxes)[0]

    # Cal iou, find the target anchor
    with tf.name_scope("Matching") as subscope:
      ious = batch_iou_fast(xywh_to_yxyx(anchors), bboxes)
      anchor_indices = tf.reshape(tf.arg_max(ious, dimension=1), shape=[-1, 1])  # target anchor indices
      # anchor_indices = tf.Print(anchor_indices, [anchor_indices], "anchor_indices", summarize=100)

      # discard duplicate # unique_idx wrong
      anchor_indices, idx, count = tf.unique_with_counts(tf.reshape(anchor_indices, shape=[-1]))
      ori_idx = tf.cumsum(tf.pad(count, [[1, 0]]))[:-1]
      anchor_indices = tf.reshape(anchor_indices, shape=[-1, 1])
      bboxes = tf.gather(bboxes, tf.unique(ori_idx)[0])
      labels = tf.gather(labels, tf.unique(ori_idx)[0])
      ious = tf.gather(ious, tf.unique(ori_idx)[0])
      num_bboxes = tf.shape(anchor_indices)[0]

      # TODO(shizehao):deal with duplicate
      # with tf.name_scope("Deal_with_duplicate"):
      #   dup_anchor_indices, indices_in_a, dup_anchor_indices_with_dup = find_dup(tf.reshape(anchor_indices, shape=[-1]))
      #
      #   # reset duplicated corresponding anchor
      #   conflicted_ious = tf.gather(ious, indices_in_a)
      #   top_k_anchor_indices = tf.nn.top_k(conflicted_ious, k=20).indices  # shape = [num_conflicted_bboxes, 20]
      #   dup_group_idx = tf.where(tf.equal(dup_anchor_indices_with_dup, tf.reshape(dup_anchor_indices, shape=[-1, 1])))
      #   seg_group = tf.unstack(dup_group_idx, axis=1)[0]


      with tf.name_scope("Deal_with_noneoverlap"):
        # find the none-overlap bbox
        bbox_indices = tf.reshape(tf.range(num_bboxes), shape=[-1, 1])
        # bbox_indices = tf.Print(bbox_indices, [bbox_indices], "bbox_indices", summarize=100)

        # anchor_indices = tf.Print(anchor_indices, [anchor_indices], "anchor_indices", summarize=100)
        iou_indices = tf.concat([bbox_indices, tf.cast(anchor_indices, dtype=tf.int32)], axis=1)
        # iou_indices = tf.Print(iou_indices, [iou_indices], "iou_indices", summarize=100)

        target_iou = tf.gather_nd(ious, iou_indices)
        # target_iou = tf.Print(target_iou,[target_iou],"target_iou",summarize=100)

        none_overlap_bbox_indices = tf.where(target_iou <= 0)  # 1-D
        # none_overlap_bbox_indices = tf.Print(none_overlap_bbox_indices, [none_overlap_bbox_indices], "none_overlap_bbox_indices", summarize=100)

        # find it's corresponding anchor
        target_bbox = tf.gather_nd(bboxes, none_overlap_bbox_indices)
        # target_bbox = tf.Print(target_bbox, [target_bbox], "target_bbox", summarize=100)

        closest_anchor_indices = arg_closest_anchor(target_bbox, xywh_to_yxyx(anchors))  # 1-D
        # closest_anchor_indices = tf.Print(closest_anchor_indices, [closest_anchor_indices, tf.gather(anchors, closest_anchor_indices)], "closest_anchor_indices", summarize=100)

      with tf.name_scope("Update_anchor_indices"):
        anchor_indices = tf.reshape(anchor_indices, shape=[-1])
        anchor_indices = update_tensor(anchor_indices, none_overlap_bbox_indices, closest_anchor_indices)
        anchor_indices = tf.reshape(anchor_indices, shape=[-1, 1])


    with tf.name_scope("Delta") as subscope:
      target_anchors = tf.gather_nd(anchors, anchor_indices)
      bboxes = yxyx_to_xywh(bboxes)
      delta = batch_delta(bboxes, target_anchors)



    with tf.name_scope("Scattering") as subscope:
      # bbox
      box_input = tf.scatter_nd(anchor_indices,
                                bboxes,
                                shape=[num_anchors, 4]
                                )

      # label
      labels_input = tf.scatter_nd(anchor_indices,
                                   tf.one_hot(labels, num_classes),
                                   shape=[num_anchors, num_classes]
                                   )

      # delta
      box_delta_input = tf.scatter_nd(anchor_indices,
                                      delta,
                                      shape=[num_anchors, 4]
                                      )





      # anchor mask
      # unique_indices, _ = tf.unique(tf.reshape(anchor_indices, shape=[-1]))
      # unique_indices = tf.Print(unique_indices, [unique_indices], summarize=100)
      # num_bboxes = tf.Print(num_bboxes, [num_bboxes])
      input_mask = tf.scatter_nd(anchor_indices,
                                 tf.ones([num_bboxes]),
                                 shape=[num_anchors])
      input_mask = tf.reshape(input_mask, shape=[-1, 1])

  return input_mask, labels_input, box_delta_input, box_input


def filter_prediction(boxes, probs, cls_idx):
  """Filter bounding box predictions with probability threshold and
  non-maximum supression.

  Args:
    boxes: array of [cx, cy, w, h].
    probs: array of probabilities
    cls_idx: array of class indices
  Returns:
    final_boxes: array of filtered bounding boxes.
    final_probs: array of filtered probabilities
    final_cls_idx: array of filtered class indices
  """
  pass
  # if config.TOP_N_DETECTION < len(probs) and config.TOP_N_DETECTION > 0:
  #   order = probs.argsort()[:-config.TOP_N_DETECTION - 1:-1]
  #   probs = probs[order]
  #   boxes = boxes[order]
  #   cls_idx = cls_idx[order]
  # else:
  #   filtered_idx = np.nonzero(probs > config.PROB_THRESH)[0]
  #   probs = probs[filtered_idx]
  #   boxes = boxes[filtered_idx]
  #   cls_idx = cls_idx[filtered_idx]
  #
  # final_boxes = []
  # final_probs = []
  # final_cls_idx = []
  #
  # for c in range(config.CLASSES):
  #   idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
  #   keep = nms(boxes[idx_per_class], probs[idx_per_class], config.NMS_THRESH)
  #   for i in range(len(keep)):
  #     if keep[i]:
  #       final_boxes.append(boxes[idx_per_class[i]])
  #       final_probs.append(probs[idx_per_class[i]])
  #       final_cls_idx.append(c)
  # return final_boxes, final_probs, final_cls_idx


def viz_prediction_result(images, batch_det_bbox, batch_det_class, batch_det_prob):
  pass
  # for i in range(len(images)):
  #   # draw prediction
  #   det_bbox, det_prob, det_class = filter_prediction(
  #     batch_det_bbox[i], batch_det_prob[i], batch_det_class[i])
  #
  #   keep_idx = [idx for idx in range(len(det_prob)) \
  #               if det_prob[idx] > config.PLOT_PROB_THRESH]
  #   det_bbox = [det_bbox[idx] for idx in keep_idx]
  #   det_prob = [det_prob[idx] for idx in keep_idx]
  #   det_class = [det_class[idx] for idx in keep_idx]
  #
  #   _draw_box(
  #       images[i], det_bbox,
  #       [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
  #           for idx, prob in zip(det_class, det_prob)],
  #       (0, 0, 255))