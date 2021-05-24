# -f*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:14:16 2021

@author: Antoine
"""

from CST.base_transformers.shapelets import Convolutional_shapelet
from CST.base_transformers.minirocket import MiniRocket
#from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from sklearn.linear_model import RidgeClassifierCV
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'Adiac', normalize=True)

#0:00:42.524087
#0:01:55.721411
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
from datetime import datetime

cst = ConvolutionalShapeletTransformer(verbose=0, random_state=0).fit(X_train, y_train)

X_cst_train = cst.transform(X_train, store=True)
X_cst_test = cst.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=400,class_weight='balanced').fit(X_cst_train, y_train)
pred = rf.predict(X_cst_test)
print("Accuracy Score for CST RF : {}".format(accuracy_score(y_test, pred)))
# In[]:
rdg = RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                        normalize=True,class_weight='balanced').fit(X_cst_train, y_train)
pred = rdg.predict(X_cst_test)
print("Accuracy Score for CST Rdg: {}".format(accuracy_score(y_test, pred)))

# In[]:
from CST.utils.shapelets_utils import generate_strides_1D
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
n_classes = np.unique(y_train).shape[0]
topk = 5
sample_id = 18
print(y_test[sample_id])
x_dist = 1/X_cst_train
i_ft = np.argsort(rf.feature_importances_)[::-1][0:topk].astype(str)
fig, ax = plt.subplots(nrows=topk, ncols=3,figsize=(5*n_classes,5*topk))
df = pd.DataFrame(x_dist)
df.columns = df.columns.astype(str)
df['y'] = y_train

for i in range(topk):
    df_long = pd.melt(df[[i_ft[i],"y"]], "y", var_name=" ", value_name="")
    sns.boxplot(x=" ", hue="y", y="", data=df_long, ax=ax[i,0], linewidth=2.5)
    ax[i,0].axhline(1/X_cst_test[sample_id, int(i_ft[i])],color='red',linestyle='--')
    ax[0,0].set_title("BoxPlot of 1/d for training samples")    
    ax[0,0].set_xlabel("Shapelet n° {}".format(i_ft[0]))
    ax[0,1].set_title("Shapelet")
    ax[0,2].set_title("Test sample with scaled shapelet")
    ax[i,2].plot(X_test[sample_id][0])
    dil = cst.dil[int(i_ft[i])]
    shp = cst.shp[int(i_ft[i])]
    x = generate_strides_1D(X_test[sample_id][0],9, dil)
    x = (x - x.mean(axis=-1, keepdims=True))/x.std(axis=-1, keepdims=True)
    d = cdist(x, shp.reshape(1,9),metric='sqeuclidean')
    loc = d.argmin()
    x_indexes = [loc + j*dil for j in range(9)]
    shp_v = (shp * X_test[sample_id, 0, x_indexes].std()) + X_test[sample_id, 0, x_indexes].mean()
    ax[i,2].scatter(x_indexes,shp_v, color='red')
    ax[i,1].scatter([0,1,2,3,4,5,6,7,8],shp, color='red')
    
    
    

