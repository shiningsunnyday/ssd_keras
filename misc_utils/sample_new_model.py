import h5py
import numpy as np
import shutil

from tensor_sampling_utils import sample_tensors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output",required=True)
parser.add_argument("--num_classes",required=True)
parser.add_argument("--source",default='base_models/VGG_coco_SSD_300x300_iter_400000.h5')
flags = parser.parse_args()
weights_source_path = flags.source
weights_destination_path = flags.output

# Make a copy of the weights file.
shutil.copy(weights_source_path, weights_destination_path)


# In[4]:


# Load both the source weights file and the copy we made.
# We will load the original weights file in read-only mode so that we can't mess up anything.
weights_source_file = h5py.File(weights_source_path, 'r')
weights_destination_file = h5py.File(weights_destination_path)

# classifier_names = ['conv4_3_norm_mbox_conf',
#                     'fc7_mbox_conf',
#                     'conv6_2_mbox_conf',
#                     'conv7_2_mbox_conf',
#                     'conv8_2_mbox_conf',
#                     'conv9_2_mbox_conf']

classifier_names = ['block_11_conf',
                    'block_9_conf',
                    'block_13_conf',
                    'block_7_conf',
                    'block_5_conf',
                    'block_4_conf']

# ## 3. Figure out which slices to pick
# 
# The following section is optional. I'll look at one classification layer and explain what we want to do, just for your understanding. If you don't care about that, just skip ahead to the next section.
# 
# We know which weight tensors we want to sub-sample, but we still need to decide which (or at least how many) elements of those tensors we want to keep. Let's take a look at the first of the classifier layers, "`conv4_3_norm_mbox_conf`". Its two weight tensors, the kernel and the bias, have the following shapes:

# In[6]:

conv4_3_norm_mbox_conf_kernel = weights_source_file['model_weights'][classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_source_file['model_weights'][classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)


# So the last axis has 324 elements. Why is that?
# 
# - MS COCO has 80 classes, but the model also has one 'backgroud' class, so that makes 81 classes effectively.
# - The 'conv4_3_norm_mbox_loc' layer predicts 4 boxes for each spatial position, so the 'conv4_3_norm_mbox_conf' layer has to predict one of the 81 classes for each of those 4 boxes.
# 
# That's why the last axis has 4 * 81 = 324 elements.
# 
# So how many elements do we want in the last axis for this layer?
# 
# Let's do the same calculation as above:
# 
# - Our dataset has 8 classes, but our model will also have a 'background' class, so that makes 9 classes effectively.
# - We need to predict one of those 9 classes for each of the four boxes at each spatial position.
# 
# That makes 4 * 9 = 36 elements.
# 
# Now we know that we want to keep 36 elements in the last axis and leave all other axes unchanged. But which 36 elements out of the original 324 elements do we want?
# 
# Should we just pick them randomly? If the object classes in our dataset had absolutely nothing to do with the classes in MS COCO, then choosing those 36 elements randomly would be fine (and the next section covers this case, too). But in our particular example case, choosing these elements randomly would be a waste. Since MS COCO happens to contain exactly the 8 classes that we need, instead of sub-sampling randomly, we'll just take exactly those elements that were trained to predict our 8 classes.
# 
# Here are the indices of the 9 classes in MS COCO that we are interested in:
# 
# `[0, 1, 2, 3, 4, 6, 8, 10, 12]`
# 
# The indices above represent the following classes in the MS COCO datasets:
# 
# `['background', 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic_light', 'stop_sign']`
# 
# How did I find out those indices? I just looked them up in the annotations of the MS COCO dataset.
# 
# While these are the classes we want, we don't want them in this order. In our dataset, the classes happen to be in the following order as stated at the top of this notebook:
# 
# `['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'traffic_light', 'motorcycle', 'bus', 'stop_sign']`
# 
# For example, '`traffic_light`' is class ID 5 in our dataset but class ID 10 in the SSD300 MS COCO model. So the order in which I actually want to pick the 9 indices above is this:
# 
# `[0, 3, 8, 1, 2, 10, 4, 6, 12]`
# 
# So out of every 81 in the 324 elements, I want to pick the 9 elements above. This gives us the following 36 indices:

# In[7]:


# n_classes_source = 81
# classes_of_interest = [0,4,]
#
# subsampling_indices = []
# for i in range(int(324/n_classes_source)):
#     indices = np.array(classes_of_interest) + i * n_classes_source
#     subsampling_indices.append(indices)
# subsampling_indices = list(np.concatenate(subsampling_indices))

# print(subsampling_indices)


# These are the indices of the 36 elements that we want to pick from both the bias vector and from the last axis of the kernel tensor.
# 
# This was the detailed example for the '`conv4_3_norm_mbox_conf`' layer. And of course we haven't actually sub-sampled the weights for this layer yet, we have only figured out which elements we want to keep. The piece of code in the next section will perform the sub-sampling for all the classifier layers.

# ## 4. Sub-sample the classifier weights
# 
# The code in this section iterates over all the classifier layers of the source weights file and performs the following steps for each classifier layer:
# 
# 1. Get the kernel and bias tensors from the source weights file.
# 2. Compute the sub-sampling indices for the last axis. The first three axes of the kernel remain unchanged.
# 3. Overwrite the corresponding kernel and bias tensors in the destination weights file with our newly created sub-sampled kernel and bias tensors.
# 
# The second step does what was explained in the previous section.
# 
# In case you want to **up-sample** the last axis rather than sub-sample it, simply set the `classes_of_interest` variable below to the length you want it to have. The added elements will be initialized either randomly or optionally with zeros. Check out the documentation of `sample_tensors()` for details.

# In[39]:


# TODO: Set the number of classes in the source weights file. Note that this number must include
#       the background class, so for MS COCO's 80 classes, this must be 80 + 1 = 81.
n_classes_source = 81
# TODO: Set the indices of the classes that you want to pick for the sub-sampled weight tensors.
#       In case you would like to just randomly sample a certain number of classes, you can just set
#       `classes_of_interest` to an integer instead of the list below. Either way, don't forget to
#       include the background class. That is, if you set an integer, and you want `n` positive classes,
#       then you must set `classes_of_interest = n + 1`.
# classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]
classes_of_interest = 16 # Uncomment this in case you want to just randomly sub-sample the last axis instead of providing a list of indices.

for name in classifier_names:
    # Get the trained weights for this layer from the source HDF5 weights file.
    kernel = weights_source_file['model_weights'][name][name]['kernel:0'].value
    bias = weights_source_file['model_weights'][name][name]['bias:0'].value

    # Get the shape of the kernel. We're interested in sub-sampling
    # the last dimension, 'o'.
    height, width, in_channels, out_channels = kernel.shape
    
    # Compute the indices of the elements we want to sub-sample.
    # Keep in mind that each classification predictor layer predicts multiple
    # bounding boxes for every spatial location, so we want to sub-sample
    # the relevant classes for each of these boxes.
    if isinstance(classes_of_interest, (list, tuple)):
        subsampling_indices = []
        for i in range(int(out_channels/n_classes_source)):
            indices = np.array(classes_of_interest) + i * n_classes_source
            subsampling_indices.append(indices)
        subsampling_indices = list(np.concatenate(subsampling_indices))
    elif isinstance(classes_of_interest, int):
        subsampling_indices = int(classes_of_interest * (out_channels/n_classes_source))
    else:
        raise ValueError("`classes_of_interest` must be either an integer or a list/tuple.")
    
    # Sub-sample the kernel and bias.
    # The `sample_tensors()` function used below provides extensive
    # documentation, so don't hesitate to read it if you want to know
    # what exactly is going on here.
    new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                          sampling_instructions=[height, width, in_channels, subsampling_indices],
                                          axes=[[3]], # The one bias dimension corresponds to the last kernel dimension.
                                          init=['gaussian', 'zeros'],
                                          mean=0.0,
                                          stddev=0.005)
    
    # Delete the old weights from the destination file.
    del weights_destination_file['model_weights'][name][name]['kernel:0']
    del weights_destination_file['model_weights'][name][name]['bias:0']
    # Create new datasets for the sub-sampled weights.
    weights_destination_file['model_weights'][name][name].create_dataset(name='kernel:0', data=new_kernel)
    weights_destination_file['model_weights'][name][name].create_dataset(name='bias:0', data=new_bias)

# Make sure all data is written to our output file before this sub-routine exits.
weights_destination_file.flush()


# That's it, we're done.
# 
# Let's just quickly inspect the shapes of the weights of the '`conv4_3_norm_mbox_conf`' layer in the destination weights file:

# In[44]:


conv4_3_norm_mbox_conf_kernel = weights_destination_file['model_weights'][classifier_names[0]][classifier_names[0]]['kernel:0']
conv4_3_norm_mbox_conf_bias = weights_destination_file['model_weights'][classifier_names[0]][classifier_names[0]]['bias:0']

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

