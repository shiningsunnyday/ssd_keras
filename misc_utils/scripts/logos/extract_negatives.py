import numpy as np
import os
from bs4 import BeautifulSoup

logos_annotations_dir = '../LogosClean/voc_format'
filenames = '../LogosClean/commonformat/ImageSets/data2.txt'
train_filenames = '../LogosClean/commonformat/ImageSets/top_only_train.txt'
test_filenames = '../LogosClean/commonformat/ImageSets/top_only_test.txt'
inpaths = [train_filenames, test_filenames]
# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = np.loadtxt("../LogosClean/commonformat/ImageSets/top_classes.txt", dtype=str)
classes = np.array([c.lower() for c in classes]).tolist()
with open("../LogosClean/classes.txt") as f:
    all_classes = np.array([line.strip() for line in f])
neg_classes = np.array([c.replace(" ","") for c in all_classes if c not in classes])
classes[-1] = 'vw'

outpaths = ['../LogosClean/commonformat/ImageSets/top_only_train_with_neg.txt',
            '../LogosClean/commonformat/ImageSets/top_only_test_with_neg.txt']

def parse_negative(filenames,
                   annotations_dir,
                   num_annotations,
                   neg_classes,
                   inpaths,
                   outpaths):
    # filenames: all filenames
    # annotations_dir: same as parse_xml
    # num_annotations: max annotations to collect, -1 means all
    # neg_classes: classes to consider as "negative"
    # outpaths: [where to append train, where to append test]
    neg_filenames = None
    if num_annotations < 0:
        num_annotations = 99999999
    np.random.seed(1)
    with open(filenames) as f:
        neg_filenames = np.array([line.strip() for line in f
                                  if line.strip().split('/')[0] in neg_classes])
    np.random.shuffle(neg_filenames)
    neg_filenames = neg_filenames[:5 * (len(neg_filenames) // 5)]
    train_neg_filenames, test_neg_filenames = neg_filenames[:int(0.8 * len(neg_filenames))], \
                                              neg_filenames[int(0.8 * len(neg_filenames)):]
    collected_count = 0
    for i in range(len(test_neg_filenames)):
        train_next, test_next = train_neg_filenames[4 * i:4 * i + 4], test_neg_filenames[i]
        for image_file in train_next:
            annotation_path = os.path.join(annotations_dir, image_file + '.xml')
            with open(annotation_path) as f:
                soup = BeautifulSoup(f, 'xml')
                collected_count += len(soup.find_all('object'))
        annotation_path = os.path.join(annotations_dir, test_next + '.xml')
        with open(annotation_path) as f:
            soup = BeautifulSoup(f, 'xml')
            collected_count += len(soup.find_all('object'))
        if collected_count > num_annotations or i==len(test_neg_filenames)-1:
            train_end, test_end = train_neg_filenames[:4 * i], test_neg_filenames[:i]
            train_current, test_current = np.loadtxt(inpaths[0], dtype=str), np.loadtxt(inpaths[1], dtype=str)
            train_output, test_output = np.append(train_current, train_end), np.append(test_current, test_end)
            np.savetxt(outpaths[0], train_output, fmt='%s')
            np.savetxt(outpaths[1], test_output, fmt='%s')
            return
    raise Exception("Number of annotations too high!")

parse_negative(filenames,logos_annotations_dir,-1,neg_classes,inpaths,outpaths)