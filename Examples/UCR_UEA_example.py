# -f*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:14:16 2021

@author: Antoine
"""

from CST.base_transformers.shapelets import Convolutional_shapelet
from CST.base_transformers.minirocket import MiniRocket
#from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
#from CST.shapelet_transforms.try_CST import ConvolutionalShapeletTransformer_tree
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
cst = ConvolutionalShapeletTransformer_tree(verbose=1, n_bins=9).fit(X_train, y_train)

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

classes = y_test
n_classes = np.unique(classes).shape[0]
lb = 0.85
coefs = rdg.coef_
if n_classes == 2:
    coefs = np.concatenate((coefs, -coefs), axis=0)

x = X_test[0][0]
x_d = np.zeros((n_classes, x.shape[0]))
x_n_shp = np.zeros((n_classes,x.shape[0]))
for i_c in range(n_classes):

    for i_shp in np.where(coefs[i_c]>=coefs[i_c].max()*lb)[0]:
        d = 0
        for dil in cst.shapelets_values:
            l = len(cst.shapelets_values[dil])
            if d<i_shp<d+l:
                shp = Convolutional_shapelet(values=cst.shapelets_values[dil][i_shp-d], dilation=int(dil), padding=0, input_ft_id=0)
                locs = shp._locate(x)
                x_n_shp[i_c, locs] += 1
                x_d[i_c, locs] += (((x[locs]-x[locs].mean()) / x[locs].std()) - shp.values)**2
            d+=l
x_d /= x_n_shp
# In[]:
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
plot_x = np.asarray(range(x.shape[0]))
plot_y = x
points = np.array([plot_x, plot_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
fig, axs = plt.subplots(ncols=n_classes, nrows=1, sharex=True, sharey=True,figsize=(15,5))
norm = plt.Normalize(x_d[x_n_shp>0].min(), x_d[x_n_shp>0].max())
for i_c in range(n_classes):
    print(i_c)
    dydx = x_d[i_c]
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(dydx)
    lc.set_linewidth(x_n_shp[i_c]*20)
    line = axs[i_c].add_collection(lc)
    fig.colorbar(line, ax=axs[i_c])
    axs[i_c].plot(plot_y)
    axs[i_c].set_xlim(plot_x.min(), plot_x.max())
    axs[i_c].set_ylim(plot_y.min(), plot_y.max())
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