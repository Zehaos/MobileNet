import tensorflow as tf
from nets.mobilenet import mobilenet
from tensorflow.python.training import saver as saver_lib
from tensorflow.python import pywrap_tensorflow

input_checkpoint = '/home/zehao/PycharmProjects/MobileNet/mobilenet-model/model.ckpt-439074'

# Where to save the modified graph
save_path = '/home/zehao/PycharmProjects/MobileNet/mobilenet-model/with_placeholder'

# TODO(shizehao): use graph editor library insead
with tf.Graph().as_default() as graph:
  input_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'MobileNet/input_images')
  logits, predictions = mobilenet(inputs=input_images, num_classes=1001, is_training=False)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    var_list = {}
    reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
      try:
        tensor = sess.graph.get_tensor_by_name(key + ":0")
      except KeyError:
        # This tensor doesn't exist in the graph (for example it's
        # 'global_step' or a similar housekeeping element) so skip it.
        continue
      var_list[key] = tensor
    saver = saver_lib.Saver(var_list=var_list)

    # Restore variables
    saver.restore(sess, input_checkpoint)

    # Save new checkpoint and the graph
    saver.save(sess, save_path+'/with_placeholder')
    tf.train.write_graph(graph, save_path, 'graph.pbtxt')


