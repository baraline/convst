# -f*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:14:16 2021

@author: Antoine
"""

# In[1]:
from CST.base_transformers.rocket import ROCKET
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
# Load GunPoint Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split('GunPoint', normalize=True)

# Init ROCKET object
rkt = ROCKET(n_kernels=10000, kernel_sizes=[9])

# Transforming data
X_rkt_train = rkt.fit_transform(X_train)
X_rkt_test = rkt.transform(X_test)

# Rocket Performance
rf = RandomForestClassifier(n_estimators=400, max_features=0.5, max_samples=0.75, ccp_alpha=0.015).fit(X_rkt_train, y_train)
pred = rf.predict(X_rkt_test)
print("F1-Score for ROCKET RF : {}".format(f1_score(y_test, pred, average='macro')))


# In[]:
    
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer

cst = ConvolutionalShapeletTransformer(rkt_object=rkt, ft_imps=rf.feature_importances_, verbose=1)
cst.fit(X_train,y_train, n_shapelet_per_combination=2, n_iter_per_comb=3, n_bins=7, percentile_select=90)
X_cst_train = cst.transform(X_train)
X_cst_test = cst.transform(X_test)

# In[]:
rf = RandomForestClassifier(n_estimators=400, ccp_alpha=0.00).fit(X_cst_train, y_train)
pred = rf.predict(X_cst_test)
print("F1-Score for CST RF : {}".format(f1_score(y_test, pred, average='macro')))
print("Used a total of {} / {} Shapelets in model".format(rf.feature_importances_[rf.feature_importances_ > 0].size, X_cst_train.shape[1]))

# In[]:
from CST.base_transformers.shapelets import Convolutional_shapelet
import numpy as np
#TODO : Adding some ploting to see some of the extracted shapelets
i_shp = np.argsort(rf.feature_importances_)[::-1][0]
#shp = Convolutional_shapelet()
grp_id = [0]
grp_id.extend(np.cumsum([len(cst.shapelets_values[k]) for k in cst.shapelets_values.keys()]))
id_grp = None
for i in range(len(grp_id)-1):
    if grp_id[i]<i_shp<grp_id[i+1]:
        id_grp = i

params = cst.shapelets_params[id_grp]
values = np.concatenate(list(cst.shapelets_values.values()),axis=0)[i_shp]
shp = Convolutional_shapelet(values=values, dilation = params[0], padding=params[2], input_ft_id=0)
# In[]:
shp.plot_loc(X_train[0,0])