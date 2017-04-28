from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from configs import kitti_config as configs


def encode_annos(image, labels, bboxes):
  """Encode annotations for losses computations.

  Args:
    image: 3-D with shape `[H, W, C]`.
    labels: 1-D with shape `[num_bounding_boxes]`.
    bboxes: 2-D with shape `[num_bounding_boxes, 4]`.

  Returns:
    input_mask: 1-D with shape `[num_anchors]`, indicate which anchor to be used to cal loss.
    labels: 2-D with shape `[num_anchors, num_classes]`, one hot encode for every anchor.
    ious: 1-D with shape `[num_anchors]`.
    box_delta_input: 2-D with shape `[num_anchors, 4]`.
  """
  #return input_mask, labels, ious, box_delta_input
  pass


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
