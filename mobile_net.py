from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def mobile_net(inputs,
          num_classes=1000,
          is_training=True,
          scope='mobile_net'):
  """MobileNet
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
  Returns:
    predictions: the softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  end_points = {}
  with tf.variable_scope(scope, 'mobile_net', [inputs]) as sc:
    with slim.arg_scope([slim.batch_norm], is_training=is_training):

      net = slim.convolution2d(inputs, 32, stride=2, padding='SAME')

      net = _depthwise_separable_conv(net, 32, 64, sc='conv_ds_1')
      net = _depthwise_separable_conv(net, 128, 128, sc='conv_ds_2')
      net = _depthwise_separable_conv(net, 128, 128, sc='conv_ds_3')
      net = _depthwise_separable_conv(net, 256, 256, sc='conv_ds_4')
      net = _depthwise_separable_conv(net, 256, 512, sc='conv_ds_5')

      net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_6')
      net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_7')
      net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_8')
      net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_9')
      net = _depthwise_separable_conv(net, 512, 512, sc='conv_ds_10')

      net = _depthwise_separable_conv(net, 512, 1024, sc='conv_ds_11')
      net = _depthwise_separable_conv(net, 1024, 1024, sc='conv_ds_12')
      net = slim.avg_pool2d(net, [7, 7], scope='avg_pool')

      logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                    scope='Logits')

      predictions = tf.nn.softmax(logits, name='Predictions')

      end_points['Logits'] = logits
      end_points['Predictions'] = predictions

      return predictions, end_points

mobile_net.default_image_size = 224


def _depthwise_separable_conv(inputs, dwc_filters, pwc_filters, sc):
  dwc = slim.separable_convolution2d(inputs, dwc_filters, kernel_size=3, activation_fn=None, scope=sc)
  bn = slim.batch_norm(dwc, scope=sc)
  relu = slim.relu(bn, scope=sc)
  pwc = slim.convolution2d(relu, pwc_filters, kernel_size=1, activation_fn=None, scope=sc)
  bn = slim.batch_norm(pwc, activation_fn=tf.nn.relu, scope=sc)
  return bn