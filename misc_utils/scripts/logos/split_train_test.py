import numpy as np
from sklearn.model_selection import train_test_split

data_path = '../LogosClean/commonformat/ImageSets/data2.txt'
classes_path = '../LogosClean/classes.txt'

data = np.loadtxt(data_path, dtype=str, delimiter='|')
classes = np.loadtxt(classes_path, dtype=str, delimiter='|')
classes = np.array([c.replace(' ', '') for c in classes])
class_to_int = dict(zip(classes, range(len(classes))))

y = []
for i in range(len(data)):
    classname = data[i].split('/')[0]
    y.append(int(class_to_int[classname]))

counts = dict(zip(np.arange(len(class_to_int)), np.array([y.count(x) for x in range(len(classes))])))
valid_keys = np.array([key for key in counts.keys() if counts[key] >= 10])



# train_data, test_data, train_y, test_y = train_test_split(X, y, test_size=0.1, stratify=y)
# print(test_data)