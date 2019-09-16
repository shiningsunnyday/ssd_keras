import numpy as np
import pandas as pd
def dist_from_arr(arr):
    dist = np.array(np.unique(arr, return_counts=True)).T[:,1]
    return dist/np.sum(dist)
def kl_divergence(dist1,dist2):
    return np.sum(dist1*np.log(dist1/dist2))


folder_path_train = '../../../../datasets/FlickrLogos_47/train/flickr_labels.csv'
folder_path_test = '../../../../datasets/FlickrLogos_47/test/flickr_labels.csv'
loc_gt_1 = pd.read_csv(folder_path_train)
loc_gt_1.loc[:,'frame'] = pd.Series(map(lambda f: 'train/'+f, loc_gt_1.loc[:,'frame']))

loc_gt_2 = pd.read_csv(folder_path_test)
loc_gt_2.loc[:,'frame'] = pd.Series(map(lambda f: 'test/'+f, loc_gt_2.loc[:,'frame']))
loc_gt = loc_gt_1.append(loc_gt_2).reset_index(drop=True)
X = loc_gt.values

# gets indices of top classes only

top_classes = np.loadtxt('../../../../datasets/FlickrLogos_47/top_classes.txt',dtype=str)
all_classes = np.loadtxt('../../../../datasets/FlickrLogos_47/className2ClassID.txt',dtype=str)
all_classes = dict(zip(all_classes[:,0], all_classes[:,1]))

top_indices = np.array([all_classes[c] for c in top_classes]).astype(int)
new_indice_map = dict(zip(top_indices, np.arange(len(top_indices))))

loc_gt = loc_gt[np.array([x in top_indices for x in X[:,-1]])].reset_index(drop=True)
X = loc_gt.values
loc_gt.class_id = np.array([new_indice_map[x] for x in X[:,-1]])

all_frames = np.array(list(dict.fromkeys(X[:,0]).keys())) # preserves order in extracting sets

divergence = 100
np.random.seed(1)
X_train,X_test = None,None

while divergence > 0.003: # forcing distribution to be very similar
    np.random.shuffle(all_frames)
    train_frames, test_frames = all_frames[:int(len(all_frames)*0.8)], all_frames[int(len(all_frames)*0.8):]
    # loc_gt_train = np.array([[X_train[i][0],int(Y_train[i]),int(X_train[i][1]),int(X_train[i][2]),int(X_train[i][3]),int(X_train[i][4])]
    #                          for i in range(X_train.shape[0])])
    # loc_gt_val = np.array([[X_test[i][0],int(Y_test[i]),int(X_test[i][1]),int(X_test[i][2]),int(X_test[i][3]),int(X_test[i][4])]
    #                        for i in range(X_test.shape[0])])
    X_train, X_test = loc_gt[np.array([loc_gt.frame[i] in train_frames for i in range(len(loc_gt))])], \
                      loc_gt[np.array([loc_gt.frame[i] in test_frames for i in range(len(loc_gt))])]
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)

    f1,f2 = np.array(X_train.class_id),np.array(X_test.class_id)
    dist1,dist2 = dist_from_arr(f1),dist_from_arr(f2)
    if dist1.size + dist2.size == 6 and (np.min(dist1) > 1e-5 and np.min(dist2) > 1e-5):
        divergence = kl_divergence(dist1,dist2)
        print(divergence)

X_train.to_csv('../../../../datasets/FlickrLogos_47/flickr_car_train_labels.csv',index=False)
X_test.to_csv('../../../../datasets/FlickrLogos_47/flickr_car_val_labels.csv',index=False)