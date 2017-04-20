# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for MobileNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import mobilenet

slim = tf.contrib.slim


class MobileNetTest(tf.test.TestCase):

  def testBuild(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, end_points = mobilenet.mobilenet(inputs, num_classes)
      self.assertEquals(end_points['MobileNet/conv_ds_2/depthwise_conv'].get_shape().as_list(), [5, 112, 112, 32])
      self.assertEquals(end_points['MobileNet/conv_ds_3/depthwise_conv'].get_shape().as_list(), [5, 56, 56, 64])
      self.assertEquals(end_points['MobileNet/conv_ds_4/depthwise_conv'].get_shape().as_list(), [5, 56, 56, 128])
      self.assertEquals(end_points['MobileNet/conv_ds_5/depthwise_conv'].get_shape().as_list(), [5, 28, 28, 128])
      self.assertEquals(end_points['MobileNet/conv_ds_6/depthwise_conv'].get_shape().as_list(), [5, 28, 28, 256])
      self.assertEquals(end_points['MobileNet/conv_ds_7/depthwise_conv'].get_shape().as_list(), [5, 14, 14, 256])
      self.assertEquals(end_points['MobileNet/conv_ds_8/depthwise_conv'].get_shape().as_list(), [5, 14, 14, 512])
      self.assertEquals(end_points['MobileNet/conv_ds_9/depthwise_conv'].get_shape().as_list(), [5, 14, 14, 512])
      self.assertEquals(end_points['MobileNet/conv_ds_10/depthwise_conv'].get_shape().as_list(), [5, 14, 14, 512])
      self.assertEquals(end_points['MobileNet/conv_ds_11/depthwise_conv'].get_shape().as_list(), [5, 14, 14, 512])
      self.assertEquals(end_points['MobileNet/conv_ds_12/depthwise_conv'].get_shape().as_list(), [5, 14, 14, 512])
      self.assertEquals(end_points['MobileNet/conv_ds_13/depthwise_conv'].get_shape().as_list(), [5, 7, 7, 512])
      self.assertEquals(end_points['MobileNet/conv_ds_14/depthwise_conv'].get_shape().as_list(), [5, 7, 7, 1024])
      self.assertEquals(end_points['squeeze'].get_shape().as_list(), [5, 1024])
      self.assertEquals(logits.op.name, 'MobileNet/fc_16/BiasAdd')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])


  def testEvaluation(self):
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = mobilenet.mobilenet(eval_inputs, is_training=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      predictions = tf.argmax(logits, 1)
      self.assertListEqual(predictions.get_shape().as_list(), [batch_size])


  def testForward(self):
    batch_size = 1
    height, width = 224, 224
    with self.test_session() as sess:
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = mobilenet.mobilenet(inputs)
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits)
      self.assertTrue(output.any())

if __name__ == '__main__':
  tf.test.main()
