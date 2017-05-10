import tensorflow as tf
from scipy.misc import imread, imresize
import numpy as np

# Quantize
use_quantized_graph = True

# Read image
img = imread("/home/zehao/Desktop/dog.png")
img = imresize(img, (224, 224, 3))
img = img.astype(np.float32)
img = np.expand_dims(img, 0)

# Preprocess
img = img / 255.
img = img - 0.5
img = img * 2.

# Graph
if use_quantized_graph:
  graph_filename = "../mobilenet-model/with_placeholder/quantized_graph.pb"
else:
  graph_filename = "../mobilenet-model/with_placeholder/frozen_graph.pb"

# Create labels dict from labels.txt
labels_file = "/home/zehao/Dataset/imagenet-data/labels.txt"
labels_dict = {}
with open(labels_file, 'r') as f:
  for kv in [d.strip().split(':') for d in f]:
    labels_dict[int(kv[0])] = kv[1]

# Create a graph def object to read the graph
with tf.gfile.GFile(graph_filename, "rb") as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())

# Construct the graph and import the graph from graphdef
with tf.Graph().as_default() as graph:
  tf.import_graph_def(graph_def)

  # We define the input and output node we will feed in
  input_node = graph.get_tensor_by_name('import/MobileNet/input_images:0')
  output_node = graph.get_tensor_by_name('import/MobileNet/Predictions/Softmax:0')

  with tf.Session() as sess:
    predictions = sess.run(output_node, feed_dict={input_node: img})[0]
    top_5_predictions = predictions.argsort()[-5:][::-1]
    top_5_probabilities = predictions[top_5_predictions]
    prediction_names = [labels_dict[i] for i in top_5_predictions]

    for i in xrange(len(prediction_names)):
      print 'Prediction: %s, Probability: %s \n' % (prediction_names[i], top_5_probabilities[i])
