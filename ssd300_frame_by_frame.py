
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from keras.optimizers import Adam, SGD
from imageio import imread
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras_ssd300_tflite_inference_pb import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss

#sample code
import argparse
from multiprocessing.pool import ThreadPool, Pool
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--model",required=True)
parser.add_argument("--input_dir",required=True)
parser.add_argument("--output_dir",required=True)
parser.add_argument("--classes",required=True)
flags = parser.parse_args()

img_height = 300
img_width = 300

K.clear_session() # Clear previous models from memory.

classes = np.loadtxt(flags.classes, dtype=str)

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=len(classes)-1,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.05, 0.10, 0.15, 0.2, 0.25, 0.37, 0.50], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
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
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=20,
                nms_max_output_size=400)

model.load_weights(flags.model, by_name=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

import os
test_folder = next(os.walk(flags.input_dir))
test_paths = [test_folder[0] + '/{}'.format(img) for img in test_folder[2]]

orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.

relevant = list(range(444,565)) + list(range(1183,1200)) + list(range(1300, 1395)) + list(range(2242, 2310)) + list(range(2808, 2970)) + list(range(3205, 3258))
for img_path in np.array(test_paths)[[x in relevant for x in np.arange(len(test_paths))]]:
    try:
        orig_images.append(imread(img_path))
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        input_images.append(img)
    except ValueError:
        pass
input_images = np.array(input_images)

confidence_threshold = 0.1
colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()

import cv2

for index in range(len(input_images)):
    y_pred = model.predict(np.expand_dims(input_images[index], axis=0))
    y_pred_thresh = [y_pred[k][y_pred[k,:,2] > confidence_threshold] for k in range(y_pred.shape[0])]
    fig, (current_axis) = plt.subplots(num=index)
    print("made fig", index)
    current_axis.imshow(orig_images[index])

    if y_pred_thresh[0].size > 0:
        box = y_pred_thresh[0][np.argmax(y_pred_thresh[0][:,2])]
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[3] * orig_images[0].shape[1] / img_width
        ymin = box[4] * orig_images[0].shape[0] / img_height
        xmax = box[5] * orig_images[0].shape[1] / img_width
        ymax = box[6] * orig_images[0].shape[0] / img_height
        color = colors[0]
        label = '{}: {:.2f}'.format(classes[int(box[1]) + 1], box[2])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-small', color='white', bbox={'facecolor': color, 'alpha': 1.0})
        print("boxes added", index)
        fig.savefig(os.path.join(flags.output_dir, 'frame{}.png'.format(index)))
        plt.close(index)
    else:
        fig.savefig(os.path.join(flags.output_dir, 'frame{}.png'.format(index)))
        plt.close(index)


def add_img(index):
    fig, (current_axis) = plt.subplots(num=index)
    print("made fig", index)
    current_axis.imshow(orig_images[index])
    for box in y_pred_thresh[index]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[3] * orig_images[0].shape[1] / img_width
        ymin = box[4] * orig_images[0].shape[0] / img_height
        xmax = box[5] * orig_images[0].shape[1] / img_width
        ymax = box[6] * orig_images[0].shape[0] / img_height
        color = colors[0]
        label = '{}: {:.2f}'.format(classes[int(box[1]) + 1], box[2])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-small', color='white', bbox={'facecolor': color, 'alpha': 1.0})
    print("boxes added", index)
    fig.savefig(os.path.join(flags.output_dir, 'frame{}.png'.format(index)))
    plt.close(index)


# poo = Pool(50)
# poo.map(add_img, list(range(len(orig_images))))

# name_map = {1:'audi',2:'bmw',3:'mercedes'}
# all_frames = []
# for i in range(len(orig_images)):
#     index = i
#     for box in y_pred_thresh[index]:
#         # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
#         xmin = box[2] * orig_images[0].shape[1] / img_width
#         ymin = box[3] * orig_images[0].shape[0] / img_height
#         xmax = box[4] * orig_images[0].shape[1] / img_width
#         ymax = box[5] * orig_images[0].shape[0] / img_height
#         all_frames.append([index, xmin, ymin, xmax, ymax, name_map[int(box[0])]])

# import pandas as pd
# pd.DataFrame(all_frames,columns=['frame','xmin','ymin','xmax','ymax','class']).to_csv('../datasets/clips/three_cars_access_frames.csv',index=False)