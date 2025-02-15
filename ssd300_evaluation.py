

from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator_custom import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_file",required=True)
parser.add_argument("--images_dir",required=True)
parser.add_argument("--labels_path",required=True)
parser.add_argument("--classes",required=True)
flags = parser.parse_args()

# In[2]:


# Set a few configuration parameters.
img_height = 300
img_width = 300
n_classes = 30
model_mode = 'inference'


# In[3]:


K.clear_session() # Clear previous models from memory.
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=n_classes,
                mode=model_mode,
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)
weights_path = flags.model_file

model.load_weights(weights_path, by_name=True)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)


# In[4]:


dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# logos_images_dir = '../datasets/LogosInTheWild-v2/LogosClean/voc_format'
# logos_annotations_dir = '../datasets/LogosInTheWild-v2/LogosClean/voc_format'
# filenames = '../datasets/LogosInTheWild-v2/LogosClean/commonformat/ImageSets/data2.txt'
# train_filenames = '../datasets/LogosInTheWild-v2/LogosClean/commonformat/ImageSets/top_only_train_with_neg.txt'
# test_filenames = '../datasets/LogosInTheWild-v2/LogosClean/commonformat/ImageSets/top_only_test_with_neg.txt'
#
# classes = np.loadtxt("../datasets/LogosInTheWild-v2/LogosClean/commonformat/ImageSets/top_classes.txt", dtype=str)
# classes = np.array([c.lower() for c in classes]).tolist()
# classes[-1] = 'vw'

classes = np.loadtxt(flags.classes,dtype=str)

dataset.parse_csv(flags.images_dir,flags.labels_path,['image_name','xmin','xmax','ymin','ymax','class_id'])


evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=1,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results

avg_prec_sorted_dic = dict(zip(classes[1:],average_precisions[1:]))
avg_prec_sorted = sorted(list(avg_prec_sorted_dic.keys()),key=lambda x:avg_prec_sorted_dic[x])

# In[ ]:



for i in range(len(avg_prec_sorted)):
    print("{:<14}{:<6}{}".format(avg_prec_sorted[i], 'AP', round(avg_prec_sorted_dic[avg_prec_sorted[i]], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

np.savetxt('../datasets/FlickrLogos_47/top_classes.txt',avg_prec_sorted[-10:],fmt='%s')

# sample code
# python ssd300_evaluation.py --model_file=models/flickr/0_layers_baseline.h5 --images_dir=../datasets/FlickrLogos_47/test --labels_path=../datasets/FlickrLogos_47/test/flickr_test_labels.csv --classes=../datasets/FlickrLogos_47/classes.txt