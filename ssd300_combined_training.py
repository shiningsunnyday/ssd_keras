# sample code
# python ssd300_combined_training.py --memory_frac=0.9 --model_file=base_models/VGG_coco_SSD_combined.h5 --saved_models=models/combined/baseline.h5 --training_summary=training_summaries/combined/baseline.csv --start_freeze=4 --end_freeze=4 --initial_epoch=0 --end_epoch=200
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--memory_frac",required=True)
parser.add_argument("--model_file",required=True)
parser.add_argument("--saved_models",required=True)
parser.add_argument("--training_summary",required=True)
parser.add_argument("--start_freeze",required=True)
parser.add_argument("--end_freeze",required=True)
parser.add_argument("--initial_epoch",required=True)
parser.add_argument("--end_epoch",required=True)
parser.add_argument("--momentum")
parser.add_argument("--fixed_lr")
parser.add_argument("--batch_size") # different batch sizes to try
parser.add_argument("--epoch_sizes")
parser.add_argument("--lr_drops") # comma separated string with  #learning rate drops values
flags = parser.parse_args() # make sure to tune learning schedule!


K.clear_session()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = float(flags.memory_frac)
sess = tf.Session(config=config)
K.set_session(sess)

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator_custom import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 30 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True


model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)


weights_path = flags.model_file

model.load_weights(weights_path, by_name=True)

sgd_momentum = 0.9 if flags.momentum is None else float(flags.momentum)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=sgd_momentum, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
freeze_range = range(int(flags.start_freeze),int(flags.end_freeze)) # layers to freeze, currently first four blocks
for i in freeze_range:
    model.layers[i].trainable = False
optimizer = sgd
model.compile(optimizer=optimizer, loss=ssd_loss.compute_loss)

train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

images_dir = '../datasets'
train_labels = '../datasets/combined_train.csv'
test_labels = '../datasets/combined_test.csv'

classes = np.loadtxt("../datasets/FlickrLogos_47/classes.txt", dtype=str)

train_dataset.parse_csv(images_dir,train_labels,['image_name','xmin','xmax','ymin','ymax','class_id'])
val_dataset.parse_csv(images_dir,test_labels,['image_name','xmin','xmax','ymin','ymax','class_id'])


# train_dataset.create_hdf5_dataset(file_path='../datasets/flickr_top_train_dataset.h5',
#                                   resize=False,
#                                   variable_image_size=True,
#                                   verbose=True)
#
# val_dataset.create_hdf5_dataset(file_path='../datasets/flickr_top_val_dataset.h5',
#                                 resize=False,
#                                 variable_image_size=True,
#                                 verbose=True)


batch_size = int(flags.batch_size)

ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=1,
                                     shuffle=True,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

lr_drops = list(map(float,flags.lr_drops.split(",")))
epoch_sizes = np.array(list(map(int,flags.epoch_sizes.split(","))))
def lr_schedule(epoch):
    return lr_drops[np.max(np.where(epoch>=epoch_sizes))]

model_checkpoint = ModelCheckpoint(filepath=flags.saved_models,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
csv_logger = CSVLogger(filename=flags.training_summary,
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]


initial_epoch   = int(flags.initial_epoch)
final_epoch     = int(flags.end_epoch)
steps_per_epoch = 100

history = model.fit_generator(generator=train_generator,
                              use_multiprocessing=True,
                              workers=64,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=1,
                              initial_epoch=initial_epoch)