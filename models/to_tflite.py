
import tensorflow as tf
from tensorflow.python.platform import gfile

# f = gfile.FastGFile("models/tflite_conversion_models/2.pb", 'rb')
# graph_def = tf.GraphDef()
# graph_def.ParseFromString(f.read())
# f.close()
#
#
# with tf.Session() as sess:
#     sess.graph.as_default()
#     tf.import_graph_def(graph_def)
#     input_tensor = sess.graph.get_tensor_by_name('input_1:0')
#     output_tensor = sess.graph.get_tensor_by_name('predictions:0')
#     sess.run(tf.global_variables_initializer())
#     tflite_converter = tf.lite.TFLiteConverter
#     tflite_converter.allow_custom_ops = True
#     tflite_converter.target_ops = set([OpsSet.TFLITE_BUILTINS, OpsSet.]))
#     converter = tflite_converter.from_session(sess, [input_tensor], [output_tensor])
#     tflite_model = converter.convert()
#     open("models/tflite_conversion_models/2.tflite", "wb").write(tflite_model)

# tflite_convert --output_file="models/tflite_conversion_models/1_w_2.tflite" --input_arrays='input_1' --output_arrays="predictions/concat" --graph_def="models/tflite_conversion_models/1_w_2.pb" --allow_custom_ops --target_ops="TFLITE_BUILTINS,SELECT_TF_OPS"