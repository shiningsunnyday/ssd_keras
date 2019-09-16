import argparse
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--classes",required=True)
parser.add_argument("--classes_id")
parser.add_argument("--datasets_dir",required=True)
parser.add_argument("--output",required=True)
parser.add_argument("--ratio",required=True)
parser.add_argument("--div",default=0.0005)

flags = parser.parse_args()
classes = [dataset.split(",") for dataset in flags.classes.split(";")]

classes_paths = [os.path.join(flags.datasets_dir, 'belgas/belgas_classes.txt'),
         os.path.join(flags.datasets_dir, 'LogosInTheWild-v2/LogosClean/commonformat/ImageSets/classes.txt'),
         os.path.join(flags.datasets_dir, 'FlickrLogos_47/classes.txt')]
gt_paths = [os.path.join(flags.datasets_dir, 'belgas/local_groundtruth.csv'),
         os.path.join(flags.datasets_dir, 'LogosInTheWild-v2/LogosClean/commonformat/ImageSets/logos_all.csv'),
         os.path.join(flags.datasets_dir, 'FlickrLogos_47/flickr_all.csv')]

indices = []

desired = pd.DataFrame()
neg = pd.DataFrame()
str_fmt = ['belgas_images/{}', 'LogosInTheWild-v2/LogosClean/voc_format/{}', 'FlickrLogos_47/{}']
for i in range(len(classes)):
    dataset_classes = np.loadtxt(classes_paths[i], dtype=str).tolist()
    indices.append([dataset_classes.index(c) for c in classes[i]])
    dataset = pd.read_csv(gt_paths[i])
    neg_dataset = dataset[[c not in indices[i] for c in dataset.class_id]]
    dataset = dataset[[c in indices[i] for c in dataset.class_id]]
    neg_dataset['frame'] = [str_fmt[i].format(f) for f in neg_dataset['frame']]
    neg_dataset['class_id'] = 0
    neg = neg.append(neg_dataset)
    dataset['frame'] = [str_fmt[i].format(f) for f in dataset['frame']]
    dataset['class_id'] = [indices[i].index(id) + sum([len(classes[j]) for j in range(i)]) for id in dataset['class_id']]
    desired = desired.append(dataset)

if flags.classes_id is not None:
    ids = list(map(int,flags.classes_id.replace(';',',').split(",")))
    id_dic = dict(zip(np.arange(len(ids)), ids))
    desired['class_id'] = [id_dic[i] for i in desired['class_id']]

desired['class_id'] = desired['class_id'] + 1

np.random.seed(1)
desired.append(neg.sample(frac=float(flags.ratio)*len(desired)/len(neg))).reset_index(drop=True).to_csv(flags.output,index=False)
print(desired['class_id'].value_counts())

def split(path, div=0.005):
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

    while divergence > div: # forcing distribution to be very similar
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

    name = path[max(path.rfind('/'),0):].strip('.csv')
    path = path[:max(path.rfind('/'),0)]
    X_train.to_csv('{}/{}_train.csv'.format(path, name),index=False)
    X_test.to_csv('{}/{}_test.csv'.format(path, name),index=False)
# import pdb
# from imageio import imread
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def get_ar_scales(path):
    imgs=pd.read_csv(path)
    dims = {}
    for img in set(imgs.frame):
        dims[img] = imread(os.path.join(flags.datasets_dir, img)).shape[:2]
    imgs['x'], imgs['y'] = [dims[img][0] for img in imgs.frame], [dims[img][1] for img in imgs.frame]
    imgs['dx'], imgs['dy'] = (imgs['xmax']-imgs['xmin'])/imgs['x'], (imgs['ymax']-imgs['ymin'])/imgs['y']
    imgs['scale'] = np.where(imgs['x']<imgs['y'],imgs['dx'],imgs['dy'])
    plt.hist(imgs['scale'],bins=100,range=(0,1))
    plt.savefig('{}_scale.png'.format(flags.output.rstrip('.csv')))

split(flags.output,div=float(flags.div))
# get_ar_scales(flags.output)