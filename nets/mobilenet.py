from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def mobilenet(inputs,
          num_classes=1000,
          is_training=True,
          scope='MobileNet'):
  """ MobileNet
  More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """

  def _depthwise_separable_conv(inputs, dwc_filters, pwc_filters, sc,  downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    _stride = 2 if downsample else 1
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=dwc_filters,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv')
    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn

  #end_points = {}
  with tf.variable_scope(scope) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        #weights_initializer=trunc_normal,
                        activation_fn=None,
                        outputs_collections=[end_points_collection]):

      net = slim.convolution2d(inputs, 32, [3, 3], stride=2, padding='SAME', scope='conv_1')
      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          activation_fn=tf.nn.relu):
        net = _depthwise_separable_conv(net, 32, 64, sc='conv_ds_2')
        net = _depthwise_separable_conv(net, 64, 128, downsample=True, sc='conv_ds_3')
        net = _depthwise_separable_conv(net, 128, 128, sc='conv_ds_4')
        net = _depthwise_separable_conv(net, 128, 128, downsample=True, sc='conv_ds_5')
        net = _depthwise_separable_conv(net, 256, 256, sc='conv_ds_6')
        net = _depthwise_separable_conv(net, 256, 512, downsample=True, sc='conv_ds_7')

        net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_8')
        net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_9')
        net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_10')
        net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_11')
        net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_12')

        net = _depthwise_separable_conv(net, 512, 1024, downsample=True, sc='conv_ds_13')
        net = _depthwise_separable_conv(net, 1024, 1024, sc='conv_ds_14')
        net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')

    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    end_points['squeeze'] = net
    logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_16')
    predictions = slim.softmax(logits, scope='Predictions')

    end_points['Logits'] = logits
    end_points['Predictions'] = predictions

  return logits, end_points

mobilenet.default_image_size = 224


def mobilenet_arg_scope(weight_decay=0.0):
  """Defines the default mobilenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.separable_convolution2d],
      kernel_size=3,
      activation_fn=None) as sc:
    return sc
