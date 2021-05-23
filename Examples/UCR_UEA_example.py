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
    'GunPoint', normalize=True)

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
ct = ConvolutionalShapeletTransformer_interpret().fit(X_train,y_train)
a = ct.transform(X_test)

# In[]:
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

n_classes = np.unique(y_test).shape[0]
i_samples = [np.random.choice(np.where(y_test==c)[0]) for c in np.unique(y_test)]
fig, axs = plt.subplots(ncols=n_classes, nrows=1, sharex=True, sharey=True,figsize=(15,5))

axs[0].set_xlim(0, X_test.shape[2])
axs[0].set_ylim(X_test[i_samples].min()-0.15, X_test[i_samples].max()+0.15)
for i_c in range(n_classes):
    x = X_test[i_samples[i_c],0]
    plot_x = np.asarray(range(x.shape[0]))
    plot_y = x
    points = np.array([plot_x, plot_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    dydx = a[i_samples[i_c]]
    norm = plt.Normalize(a[i_samples[i_c]].min(), a[i_samples[i_c]].max())
    lc = LineCollection(segments, cmap='jet', norm=norm,alpha=0.75)
    lc.set_array(dydx)
    lw = ((dydx - dydx.min())/(dydx.max()-dydx.min()))*25
    lc.set_linewidth(lw)
    line = axs[i_c].add_collection(lc)
    fig.colorbar(line, ax=axs[i_c])
    axs[i_c].plot(plot_y,c='black',linewidth=1)    
plt.show()


"""
for i in i_shp:
    d = 0
    for dil in cst.shapelets_values:
        l = len(cst.shapelets_values[dil])
        if d<i<d+l:
            print(cst.shapelets_values[dil][i-d])
            shp = Convolutional_shapelet(values=cst.shapelets_values[dil][i-d], dilation=int(dil), padding=0, input_ft_id=0)
            fig, ax = plt.subplots(ncols=n_classes,figsize=(n_classes*5,6))
            for j in range(n_classes):
                idx = np.where(y_test==j)[0][0]
                shp.plot_loc(X_test[idx,0],ax=ax[j])
                ax[j].set_title(X_cst_test[idx,i])
            plt.show()
            break
        else:
            d+=l
"""      