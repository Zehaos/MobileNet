import tensorflow as tf
import time
from datetime import datetime
import math
import argparse
import sys
from nets.mobilenet import mobilenet, mobilenet_arg_scope

slim = tf.contrib.slim


def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0

  for i in range(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration

  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


def run_benchmark():
  with tf.Graph().as_default():
    #with tf.device('/gpu:0'):
    with tf.device('/cpu:0'):
      # Generate some dummy images.
      image_size = 224
      inputs = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                             image_size,
                                             image_size, 3],
                                            dtype=tf.float32,
                                            stddev=1e-1))

      with slim.arg_scope(mobilenet_arg_scope()):
        logits, end_points = mobilenet(inputs, is_training=False)

      init = tf.global_variables_initializer()
      config = tf.ConfigProto()
      config.gpu_options.allocator_type = 'BFC'
      sess = tf.Session(config=config)
      sess.run(init)

      time_tensorflow_run(sess, logits, "Forward")

      # Add a simple objective so we can calculate the backward pass.
      objective = tf.nn.l2_loss(logits)

      # Compute the gradient with respect to all the parameters.
      grad = tf.gradients(objective, tf.trainable_variables())

      # Run the backward benchmark.
      time_tensorflow_run(sess, grad, "Forward-backward")

def main(_):
  run_benchmark()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='Batch size.'
  )
  parser.add_argument(
    '--num_batches',
    type=int,
    default=300,
    help='Number of batches to run.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
