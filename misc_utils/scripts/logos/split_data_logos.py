import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pdb

loc_gt = np.loadtxt('../../../../datasets/LogosInTheWild-v2/LogosClean/commonformat/ImageSets/car_only.txt', dtype=str)
np.random.seed(1)
X = loc_gt

def dist_from_arr(arr):
    dist = np.array(np.unique(arr, return_counts=True)).T[:,1]
    return dist/np.sum(dist)
def kl_divergence(dist1,dist2):
    return np.sum(dist1*np.log(dist1/dist2))

# all_frames = np.array(list(dict.fromkeys(X[:,0]).keys())) # preserves order in extracting sets
all_frames = X
all_classes = np.array(list(map(lambda x: x.split('/')[0], X)))
set_classes = dict(zip(sorted(list(set(all_classes))), np.arange(len(set(all_classes)))))
all_classes = np.array([set_classes[all_classes[i]] for i in range(len(all_classes))])

divergence = 100
max_diff = 1
X_train, X_test = None, None

while divergence > 0.005: # forcing distribution to be very similar
    perm = np.random.permutation(np.arange(len(all_classes)))
    all_frames, all_classes = all_frames[perm], all_classes[perm]

    train_frames, test_frames = all_frames[:int(len(all_frames)*0.8)], all_frames[int(len(all_frames)*0.8):]
    X_train, X_test = train_frames, test_frames
    train_classes, test_classes = all_classes[:int(len(all_classes)*0.8)], all_classes[int(len(all_classes)*0.8):]
    dist1, dist2 = dist_from_arr(train_classes), dist_from_arr(test_classes)
    if np.min(dist1)<1e-5 or np.min(dist2)<1e-5:
        pass
    elif dist1.size == dist2.size:
        divergence = kl_divergence(dist1, dist2)
        print(divergence)

# X_train.to_csv('../../datasets/train_.csv',index=False)
# X_test.to_csv('../../datasets/test_.csv',index=False)

np.savetxt('../../../../datasets/LogosInTheWild-v2/LogosClean/commonformat/ImageSets/car_only_train.txt', X_train, fmt='%s')
np.savetxt('../../../../datasets/LogosInTheWild-v2/LogosClean/commonformat/ImageSets/car_only_test.txt', X_test, fmt='%s')