import tensorflow as tf
import time
from datetime import datetime
import math
import argparse
import sys
from nets.mobilenet import mobilenet, mobilenet_arg_scope
import numpy as np

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

def time_tensorflow_run_placeholder(session, target, feed_dict, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0

  for i in range(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target,feed_dict=feed_dict)
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
  if FLAGS.quantized:
    graph_filename = FLAGS.quantized_graph
    # Create a graph def object to read the graph
    with tf.gfile.GFile(graph_filename, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    with tf.device('/'+FLAGS.mode+':0'):
      image_size = 224
      if FLAGS.quantized:
        inputs = np.random.random((FLAGS.batch_size, image_size, image_size, 3))
        tf.import_graph_def(graph_def)
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)
        # We define the input and output node we will feed in
        input_node = graph.get_tensor_by_name('import/MobileNet/input_images:0')
        output_node = graph.get_tensor_by_name('import/MobileNet/Predictions/Softmax:0')
        time_tensorflow_run_placeholder(sess, output_node, {input_node: inputs}, "Forward")
      else:
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
    default=100,
    help='Number of batches to run.'
  )
  parser.add_argument(
    '--mode',
    type=str,
    default='cpu',
    help='gpu/cpu mode.'
  )
  parser.add_argument(
    '--quantized',
    type=bool,
    default=False,
    help='Benchmark quantized graph.'
  )
  parser.add_argument(
    '--quantized_graph',
    type=str,
    default='',
    help='Path to quantized graph file.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
