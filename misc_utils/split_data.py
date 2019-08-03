import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

loc_gt = pd.read_csv('../datasets/local_groundtruth.csv').values
X, Y = loc_gt, loc_gt[:,5]-1
X[:,5] = X[:,5]-1
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
# loc_gt_train = np.array([[X_train[i][0],int(Y_train[i]),int(X_train[i][1]),int(X_train[i][2]),int(X_train[i][3]),int(X_train[i][4])]
#                          for i in range(X_train.shape[0])])
# loc_gt_val = np.array([[X_test[i][0],int(Y_test[i]),int(X_test[i][1]),int(X_test[i][2]),int(X_test[i][3]),int(X_test[i][4])]
#                        for i in range(X_test.shape[0])])
train_df,test_df=pd.DataFrame(X_train),pd.DataFrame(X_test)
train_df.to_csv('../datasets/local_groundtruth_train.csv',index=False)
test_df.to_csv('../datasets/local_groundtruth_test.csv',index=False)