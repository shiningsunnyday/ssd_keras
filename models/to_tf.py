from keras.models import load_model
import tensorflow as tf
import keras.backend as K
import argparse
from keras_ssd300_tflite_inference_pb import ssd_300

parser = argparse.ArgumentParser()
parser.add_argument("--model_file",required=True)
parser.add_argument("--output_dir",required=True)
flags = parser.parse_args()

img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 3 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# scales = [0.05, 0.10, 0.15, 0.25, 0.37, 0.54, 0.88]
scales = [0.05, 0.10, 0.15, 0.2, 0.25, 0.37, 0.50]
aspect_ratios = [[1.0, 1.5, 2.0/3.0],
                 [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                 [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                 [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                 [1.0, 1.5, 2.0/3.0],
                 [1.0, 1.5, 2.0/3.0]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
steps=[8, 16, 32, 64, 100, 300]

model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='inference',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                offsets=offsets,
                steps=steps,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels,
                confidence_thresh=0.5,
                top_k=20)

name = flags.model_file.split('/')[-1]
model.load_weights(flags.model_file,by_name=True)


print(model.outputs)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        for o in output_names:
            print(o)

        print("\n", output_names[0])
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "models/tflite_conversion_models", name.replace('.h5','.pb'), as_text=False)