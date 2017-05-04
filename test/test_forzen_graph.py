import tensorflow as tf
from scipy.misc import imread, imresize
import numpy as np

img = imread("/home/zehao/Desktop/dog.png")
img = imresize(img, (224,224,3))
img = img.astype(np.float32)
img = np.expand_dims(img, 0)


#Define the filename of the frozen graph
graph_filename = "./frozen_model.pb"

#Create a graph def object to read the graph
with tf.gfile.GFile(graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

#Construct the graph and import the graph from graphdef
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

    #We define the input and output node we will feed in
    input_node = graph.get_tensor_by_name('FIFOQueueV2:0')
    output_node = graph.get_tensor_by_name('import/InceptionResnetV2/Logits/Predictions:0')

    with tf.Session() as sess:
        predictions = sess.run(output_node, feed_dict = {input_node: img})
        print predictions
        label_predicted = np.argmax(predictions[0])

    print 'Predicted Flower:', labels_dict[label_predicted]
    print 'Prediction probability:', predictions[0][label_predicted]