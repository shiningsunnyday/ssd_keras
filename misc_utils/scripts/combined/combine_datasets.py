import pandas as pd
import numpy as np

belga_train_path = '../../../../datasets/belgas/belgas_cars_train.csv'
logos_train_path = '../../../../datasets/LogosInTheWild-v2/LogosClean/commonformat/ImageSets/logos_car_train.csv'
flickr_train_path = '../../../../datasets/FlickrLogos_47/flickr_car_train_labels.csv'
belga_test_path = '../../../../datasets/belgas/belgas_cars_test.csv'
logos_test_path = '../../../../datasets/LogosInTheWild-v2/LogosClean/commonformat/ImageSets/logos_car_test.csv'
flickr_test_path = '../../../../datasets/FlickrLogos_47/flickr_car_val_labels.csv'

belga_train_csv = pd.read_csv(belga_train_path)
belga_train_csv['frame'] = pd.Series(['belgas_images/{}'.format(frame) for frame in belga_train_csv['frame']])
logos_train_csv = pd.read_csv(logos_train_path)
logos_train_csv['frame'] = pd.Series(['LogosInTheWild-v2/LogosClean/voc_format/{}'.format(frame) for frame in logos_train_csv['frame']])
flickr_train_csv = pd.read_csv(flickr_train_path)
flickr_train_csv['frame'] = pd.Series(['FlickrLogos_47/{}'.format(frame) for frame in flickr_train_csv['frame']])
belga_test_csv = pd.read_csv(belga_test_path)
belga_test_csv['frame'] = pd.Series(['belgas_images/{}'.format(frame) for frame in belga_test_csv['frame']])
logos_test_csv = pd.read_csv(logos_test_path)
logos_test_csv['frame'] = pd.Series(['LogosInTheWild-v2/LogosClean/voc_format/{}'.format(frame) for frame in logos_test_csv['frame']])
flickr_test_csv = pd.read_csv(flickr_test_path)
flickr_test_csv['frame'] = pd.Series(['FlickrLogos_47/{}'.format(frame) for frame in flickr_test_csv['frame']])

belga_train_csv['class_id'] = belga_train_csv['class_id'] + np.max(logos_train_csv['class_id']) + 1
flickr_train_csv['class_id'] = flickr_train_csv['class_id'] + np.max(belga_train_csv['class_id']) + 1
belga_test_csv['class_id'] = belga_test_csv['class_id'] + np.max(logos_test_csv['class_id']) + 1
flickr_test_csv['class_id'] = flickr_test_csv['class_id'] + np.max(belga_test_csv['class_id']) + 1

assert np.max(flickr_train_csv['class_id']) == np.max(flickr_test_csv['class_id'])

train_combined = logos_train_csv.append(belga_train_csv).append(flickr_train_csv).reset_index(drop=True)
test_combined = logos_test_csv.append(belga_test_csv).append(flickr_test_csv).reset_index(drop=True)

repeat_indices = {i:i for i in range(len())
train_combined.class_id = np.array([repeat_indices[id] for id in train_combined.class_id])
test_combined.class_id = np.array([repeat_indices[id] for id in test_combined.class_id])

train_combined.to_csv('../../../../datasets/combined_cars_train.csv',index=False)
test_combined.to_csv('../../../../datasets/combined_cars_test.csv',index=False)