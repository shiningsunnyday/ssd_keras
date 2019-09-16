
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam, SGD
from imageio import imread
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from models.keras_ssd300_tflite_mobilenet import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetectionsCustom import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
import argparse

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 300
img_width = 300
model_mode = 'inference'

# python ssd300_inference.py --model_file=models/combined/allstar_pos_only.h5 --images_dir=../datasets --output_dir=examples/allstar_pos_only --labels_path=../datasets/allstar_pos_only_test.csv --classes=../datasets/allstar_pos_only_test.csv --conf_thresh=0.5

parser = argparse.ArgumentParser()
parser.add_argument("--model_file",required=True)
parser.add_argument("--images_dir",required=True)
parser.add_argument("--output_dir",required=True)
parser.add_argument("--labels_path",required=True)
parser.add_argument("--classes",required=True)
parser.add_argument("--conf_thresh",required=True)
flags = parser.parse_args()
K.clear_session() # Clear previous models from memory.


classes = np.loadtxt(flags.classes, dtype=str).tolist()
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=len(classes)-1,
                mode=model_mode,
                l2_regularization=0.0005,
                scales=[0.05, 0.10, 0.15, 0.2, 0.25, 0.37, 0.50], # The scales for MS COCO [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 1.5, 2.0/3.0],
                 [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                 [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                 [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                 [1.0, 1.5, 2.0/3.0],
                 [1.0, 1.5, 2.0/3.0]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.5,
                top_k=200,
                nms_max_output_size=400)

weights_path = flags.model_file
model.load_weights(weights_path, by_name=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)


dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
images_dir = flags.images_dir
labels_file = flags.labels_path

dataset.parse_csv(images_dir,labels_file,['image_name','xmin','xmax','ymin','ymax','class_id'])

# TODO: Set the paths to the datasets here.

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

import tensorflow as tf
tf.random.set_random_seed(1)


# In[ ]:


# Generate a batch and make predictions.

import time

generator = dataset.generate(batch_size=1,
                                 shuffle=False,
                                 transformations=[convert_to_3_channels,
                                                  resize],
                                 returns={'processed_images',
                                          'filenames',
                                          'inverse_transform',
                                          'original_images',
                                          'original_labels'},
                                 keep_images_without_gt=False)
start_time = time.time()
for i in range(32):
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(generator)
    confidence_threshold = float(flags.conf_thresh)
    # Predict.

    y_pred = model.predict(batch_images)
end_time = time.time()

print("Average fps was", 32 / (end_time - start_time))

# # Perform confidence thresholding.
# y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
#
# # Convert the predictions for the original image.
# y_pred_thresh_inv = apply_inverse_transforms(y_pred_thresh, batch_inverse_transforms)
#
# high_conf_indices = np.array(range(y_pred.shape[0]))[[np.array([y_pred_thresh[i].size > 0 for i in range(y_pred.shape[0])])]]
#
#
# # In[ ]:
#
#
# for i in high_conf_indices:
#     # Display the image and draw the predicted boxes onto it.
#     # Set the colors for the bounding boxes
#     colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()
#     plt.close()
#     plt.figure(figsize=(20,12))
#     plt.imshow(batch_original_images[i])
#
#     current_axis = plt.gca()
#
#     for box in batch_original_labels[i]:
#         xmin = box[1]
#         ymin = box[2]
#         xmax = box[3]
#         ymax = box[4]
#         label = '{}'.format(classes[int(box[0])])
#         current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
#         current_axis.text(xmin, ymin, label, size='small', color='white', bbox={'facecolor':'green', 'alpha':1.0})
#
#     for box in y_pred_thresh_inv[i]:
#         xmin = box[2]
#         ymin = box[3]
#         xmax = box[4]
#         ymax = box[5]
#         color = colors[int(box[0])]
#         label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#         current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
#         current_axis.text(xmin, ymin, label, size='small', color='white', bbox={'facecolor':color, 'alpha':1.0})
#
#     plt.savefig('{}/inference_test_{}.png'.format(flags.output_dir, i))
#
#
# # In[ ]:




