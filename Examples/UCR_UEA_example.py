# -f*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:14:16 2021

@author: Antoine
"""

from CST.base_transformers.shapelets import Convolutional_shapelet
from CST.base_transformers.minirocket import MiniRocket
#from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from CST.shapelet_transforms.try_CST import ConvolutionalShapeletTransformer_tree
from sklearn.linear_model import RidgeClassifierCV
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'GunPoint', normalize=True)

# In[]:

# Init ROCKET object
rkt = MiniRocket()

# Transforming data
X_rkt_train = rkt.fit_transform(X_train)
X_rkt_test = rkt.transform(X_test)

# Rocket Performance
rdg = RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                        normalize=True,class_weight='balanced').fit(X_rkt_train, y_train)
pred = rdg.predict(X_rkt_test)
print("Accuracy Score for MINI-ROCKET: {}".format(accuracy_score(y_test, pred)))

# In[]:
cst = ConvolutionalShapeletTransformer_tree(verbose=1, n_bins=11).fit(X_train, y_train)
X_cst_train = cst.transform(X_train)
X_cst_test = cst.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=400,class_weight='balanced').fit(X_cst_train, y_train)
pred = rf.predict(X_cst_test)
print("Accuracy Score for CST RF : {}".format(accuracy_score(y_test, pred)))

rdg = RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                        normalize=True,class_weight='balanced').fit(X_cst_train, y_train)
pred = rdg.predict(X_cst_test)
print("Accuracy Score for CST Rdg: {}".format(accuracy_score(y_test, pred)))
# In[]:

from matplotlib import pyplot as plt

i_shp = np.argsort(rf.feature_importances_)[::-1][0:5]
for i in i_shp:
    d = 0
    for dil in cst.shapelets_values:
        l = len(cst.shapelets_values[dil])
        if d<i<d+l:
            print(cst.shapelets_values[dil][i-d])
            shp = Convolutional_shapelet(values=cst.shapelets_values[dil][i-d], dilation=int(dil), padding=0, input_ft_id=0)
            fig, ax = plt.subplots(ncols=2)
            shp.plot_loc(X_test[0,0],ax=ax[0])
            ax[0].set_title(X_cst_test[0,i])
            shp.plot_loc(X_test[1,0],ax=ax[1])
            ax[1].set_title(X_cst_test[1,i])
            plt.show()
            break
        else:
            d+=l
        