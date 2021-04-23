# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:47:27 2021

@author: A694772
"""

from CST.base_transformers.minirocket import MiniRocket
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
from CST.utils.dataset_utils import load_sktime_dataset_split
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")

# Load GunPoint Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split('GunPoint', normalize=True)

mCST = MiniConvolutionalShapeletTransformer()

locs, _, _ = mCST._init_kernels(X_train, y_train)

# In[]:
i_kernel = 3
class_locs = np.zeros((np.unique(y_train).shape[0],X_train.shape[2]))
fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15,5))
for c in np.unique(y_train):
    class_loc = np.sum(locs[np.where(y_train==c)[0],i_kernel],axis=0)
    #class_loc = (class_loc - class_loc.min())/(class_loc.max() - class_loc.min())
    class_locs[c,:] = class_loc
    ax[c].plot(class_loc, c='green')
    ax[c].plot(X_train[np.random.choice(np.where(y_train==c)[0]),0],
               color='orange' if c==1 else 'blue')
    ax[c].set_title('Normalized FIL of class {}'.format(c))
    

ax[2].plot(X_train[np.random.choice(np.where(y_train==0)[0]),0],c='blue',label='class 0')
ax[2].plot(X_train[np.random.choice(np.where(y_train==1)[0]),0],c='orange',label='class 1')
ax[2].plot(class_locs[0] - class_locs[1],c='green',label='FIL')
ax[2].set_title('FIL difference')
plt.legend()
plt.show()