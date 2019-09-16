import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split(path, name, div=0.005):
    loc_gt = pd.read_csv(path)
    np.random.seed(1)
    X = loc_gt.values

    def dist_from_arr(arr):
        dist = np.array(np.unique(arr, return_counts=True)).T[:,1]
        return dist/np.sum(dist)

    def kl_divergence(dist1,dist2):
        return np.sum(dist1*np.log(dist1/dist2))

    all_frames = np.array(list(dict.fromkeys(X[:,0]).keys())) # preserves order in extracting sets
    divergence = 100
    X_train,X_test = None,None

    while divergence > 0.0005: # forcing distribution to be very similar
        np.random.shuffle(all_frames)
        train_frames, test_frames = all_frames[:int(len(all_frames)*0.8)], all_frames[int(len(all_frames)*0.8):]
        # loc_gt_train = np.array([[X_train[i][0],int(Y_train[i]),int(X_train[i][1]),int(X_train[i][2]),int(X_train[i][3]),int(X_train[i][4])]
        #                          for i in range(X_train.shape[0])])
        # loc_gt_val = np.array([[X_test[i][0],int(Y_test[i]),int(X_test[i][1]),int(X_test[i][2]),int(X_test[i][3]),int(X_test[i][4])]
        #                        for i in range(X_test.shape[0])])
        X_train, X_test = loc_gt[np.array([loc_gt.frame[i] in train_frames for i in range(len(loc_gt))])], \
                          loc_gt[np.array([loc_gt.frame[i] in test_frames for i in range(len(loc_gt))])]
        X_train, X_test = X_train.reindex(), X_test.reindex()

        f1,f2 = np.array(X_train.class_id),np.array(X_test.class_id)
        try:
            dist1,dist2=dist_from_arr(f1),dist_from_arr(f2)
            if len(dist1) == len(dist2) and np.min(dist1) > 1e-5 and np.min(dist2) > 1e-5:
                divergence = kl_divergence(dist1,dist2)
                print(divergence)
        except ValueError:
            pass

    print(dist1, dist2)
    X_train.to_csv('../../../../datasets/{}_train.csv'.format(name),index=False)
    X_test.to_csv('../../../../datasets/{}_test.csv'.format(name),index=False)