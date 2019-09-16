# sample code

# python ssd300_tflite_training.py --memory_frac=0.3 --model_file=models/momentum_0.h5 --saved_models=models/combined/momentum_2.h5 --training_summary=training_summaries/combined/momentum_0_2.csv --start_freeze=4 --end_freeze=4 --initial_epoch=0 --end_epoch=150 --batch_size=32 --momentum=0.2 --epoch_sizes="0" --lr_drops="1e-4"

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
parser.add_argument("--alpha",default=1.0)
parser.add_argument("--neg_pos_ratio",default=3.0)
parser.add_argument("--conf_thresh",default=0.01)
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
import pandas as pd
from matplotlib import pyplot as plt

from models.keras_ssd512_tflite import ssd_512
from keras_loss_function.keras_ssd_loss_balanced import SSDLoss
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


img_height = 512 # Height of the model input images
img_width = 512 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 3 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales = [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.40, 0.50]
aspect_ratios = [[1.0, 1.5, 2.0/3.0],
                 [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                 [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                 [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                    [1.0, 1.5, 2.0/3.0, 1.25, 0.8],
                 [1.0, 1.5, 2.0/3.0],
                 [1.0, 1.5, 2.0/3.0]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
steps = [8, 16, 32, 64, 128, 256, 512]

train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)
images_dir = '../datasets'
train_labels = '../datasets/best_cars_only_edit_train.csv'
test_labels = '../datasets/best_cars_only_edit_test.csv'


train_dataset.parse_csv(images_dir,train_labels,['image_name','xmin','xmax','ymin','ymax','class_id'])
val_dataset.parse_csv(images_dir,test_labels,['image_name','xmin','xmax','ymin','ymax','class_id'])


model = ssd_512(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
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
                swap_channels=swap_channels)


weights_path = flags.model_file

model.load_weights(weights_path, by_name=True)

sgd_momentum = 0 if flags.momentum is None else float(flags.momentum)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=sgd_momentum, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
freeze_range = range(int(flags.start_freeze),int(flags.end_freeze)) # layers to freeze, currently first four blocks
for i in freeze_range:
    model.layers[i].trainable = False
optimizer = sgd
model.compile(optimizer=optimizer, loss=ssd_loss.compute_loss)

n_neg_min_ = 0

def smooth_L1_loss(y_true, y_pred):
    '''
    Compute smooth L1 loss, see references.

    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
            contains the ground truth bounding box coordinates, where the last dimension
            contains `(xmin, xmax, ymin, ymax)`.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box coordinates.

    Returns:
        The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).

    References:
        https://arxiv.org/abs/1504.08083
    '''
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)
def log_loss(y_true, y_pred):
    '''
    Compute the softmax log loss.

    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape (batch_size, #boxes, #classes)
            and contains the ground truth bounding box categories.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box categories.

    Returns:
        The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).
    '''
    # # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
    y_pred = tf.maximum(y_pred, 1e-15)
    y_weights = tf.reduce_sum(y_true, axis=1, keepdims=True)  # (batch_size, 1, #classes)
    y_weights = tf.reduce_sum(y_weights, axis=2, keepdims=True) - y_weights  # (batch_size, 1, #neg classes)
    y_weights = tf.to_float(tf.shape(y_true)[2]) * y_weights / tf.reduce_sum(y_weights, axis=-1, keepdims=True)
    # # Compute the log loss
    log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
    return log_loss
def class_loss(y_true, y_pred):
    neg_pos_ratio = tf.constant(int(flags.neg_pos_ratio))
    n_neg_min = tf.constant(n_neg_min_)
    alpha = tf.constant(float(flags.alpha))

    batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
    n_boxes = tf.shape(y_pred)[1]
    classification_loss = tf.to_float(
        log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]))  # Output shape: (batch_size, n_boxes)
    localization_loss = tf.to_float(
        smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]))  # Output shape: (batch_size, n_boxes)

    negatives = y_true[:, :, 0]  # Tensor of shape (batch_size, n_boxes)
    positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))  # Tensor of shape (batch_size, n_boxes)

    # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
    n_positive = tf.reduce_sum(positives)

    pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

    neg_class_loss_all = classification_loss * negatives  # Tensor of shape (batch_size, n_boxes)
    n_neg_losses = tf.count_nonzero(neg_class_loss_all,
                                    dtype=tf.int32)  # The number of non-zero loss entries in `neg_class_loss_all`
    n_negative_keep = tf.minimum(tf.maximum(neg_pos_ratio * tf.to_int32(n_positive), n_neg_min), n_neg_losses)

    def f1():
        return tf.zeros([batch_size])

    # Otherwise compute the negative loss.
    def f2():
        # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
        # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
        # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

        # To do this, we reshape `neg_class_loss_all` to 1D...
        neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])  # Tensor of shape (batch_size * n_boxes,)
        # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
        values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                      k=n_negative_keep,
                                      sorted=False)  # We don't need them sorted.
        # ...and with these indices we'll create a mask...
        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                       updates=tf.ones_like(indices, dtype=tf.int32),
                                       shape=tf.shape(neg_class_loss_all_1D))  # Tensor of shape (batch_size * n_boxes,)
        negatives_keep = tf.to_float(
            tf.reshape(negatives_keep, [batch_size, n_boxes]))  # Tensor of shape (batch_size, n_boxes)
        # ...and use it to keep only those boxes and mask all other classification losses
        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)  # Tensor of shape (batch_size,)
        return neg_class_loss

    neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

    class_loss = pos_class_loss + neg_class_loss  # Tensor of shape (batch_size,)

    # 3: Compute the localization loss for the positive targets.
    #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).

    loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

    # 4: Compute the total loss.

    total_loss = (class_loss + alpha * loc_loss) / tf.maximum(1.0, n_positive)  # In case `n_positive == 0`
    # with open('loss_test.txt',mode='ab') as file:
    #     np.save(file, [total_loss, class_loss/tf.maximum(1.0,n_positive), loc_loss/tf.maximum(1.0,n_positive)])
    # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
    # because the relevant criterion to average our loss over is the number of positive boxes in the batch
    # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
    # over the batch size, we'll have to multiply by it.
    total_loss = total_loss * tf.to_float(batch_size)

    return (tf.to_float(batch_size)*class_loss)/((1 + alpha) * tf.maximum(1.0, n_positive))
def loc_loss(y_true, y_pred):
    neg_pos_ratio = tf.constant(int(flags.neg_pos_ratio))
    n_neg_min = tf.constant(n_neg_min_)
    alpha = tf.constant(float(flags.alpha))

    batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
    n_boxes = tf.shape(y_pred)[1]
    classification_loss = tf.to_float(
        log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]))  # Output shape: (batch_size, n_boxes)
    localization_loss = tf.to_float(
        smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]))  # Output shape: (batch_size, n_boxes)

    negatives = y_true[:, :, 0]  # Tensor of shape (batch_size, n_boxes)
    positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))  # Tensor of shape (batch_size, n_boxes)

    # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
    n_positive = tf.reduce_sum(positives)

    pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

    neg_class_loss_all = classification_loss * negatives  # Tensor of shape (batch_size, n_boxes)
    n_neg_losses = tf.count_nonzero(neg_class_loss_all,
                                    dtype=tf.int32)  # The number of non-zero loss entries in `neg_class_loss_all`
    n_negative_keep = tf.minimum(tf.maximum(neg_pos_ratio * tf.to_int32(n_positive), n_neg_min), n_neg_losses)

    def f1():
        return tf.zeros([batch_size])

    # Otherwise compute the negative loss.
    def f2():
        # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
        # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
        # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

        # To do this, we reshape `neg_class_loss_all` to 1D...
        neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])  # Tensor of shape (batch_size * n_boxes,)
        # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
        values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                      k=n_negative_keep,
                                      sorted=False)  # We don't need them sorted.
        # ...and with these indices we'll create a mask...
        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                       updates=tf.ones_like(indices, dtype=tf.int32),
                                       shape=tf.shape(neg_class_loss_all_1D))  # Tensor of shape (batch_size * n_boxes,)
        negatives_keep = tf.to_float(
            tf.reshape(negatives_keep, [batch_size, n_boxes]))  # Tensor of shape (batch_size, n_boxes)
        # ...and use it to keep only those boxes and mask all other classification losses
        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)  # Tensor of shape (batch_size,)
        return neg_class_loss

    neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

    class_loss = pos_class_loss + neg_class_loss  # Tensor of shape (batch_size,)

    # 3: Compute the localization loss for the positive targets.
    #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).

    loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

    # 4: Compute the total loss.

    total_loss = (class_loss + alpha * loc_loss) / tf.maximum(1.0, n_positive)  # In case `n_positive == 0`
    # with open('loss_test.txt',mode='ab') as file:
    #     np.save(file, [total_loss, class_loss/tf.maximum(1.0,n_positive), loc_loss/tf.maximum(1.0,n_positive)])
    # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
    # because the relevant criterion to average our loss over is the number of positive boxes in the batch
    # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
    # over the batch size, we'll have to multiply by it.
    total_loss = alpha * total_loss * tf.to_float(batch_size)

    return (tf.to_float(batch_size)*loc_loss)/((1 + alpha) * tf.maximum(1.0, n_positive))

model.compile(optimizer=optimizer, loss=ssd_loss.compute_loss, metrics=['accuracy', class_loss, loc_loss])

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
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

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