# -*- coding: utf-8 -*-


from CST.base_transformers.convolutional_kernels import Rocket_kernel
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
from CST.utils.dataset_utils import load_sktime_dataset_split
from CST.utils.shapelets_utils import compute_distances, generate_strides_2D, generate_strides_1D
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")

# Load GunPoint Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split('GunPoint', normalize=True)

mCST = MiniConvolutionalShapeletTransformer()

locs, dils, biases = mCST._init_kernels(X_train, y_train)
# In[]:
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(15,5))
for i in range(X_train.shape[0]):
    if y_train[i] == 1:
        ax[1].plot(X_train[i,0], c='orange', alpha=0.2)
    else:
        ax[0].plot(X_train[i,0], c='blue', alpha=0.2)
ax[0].set_title("Class 0 of GunPoint")
ax[1].set_title("Class 1 of GunPoint")
plt.show()
# In[]:
i_kernel = 1

k = Rocket_kernel(length=5, bias=biases[i_kernel], dilation=dils[i_kernel], padding=0, weights=[1,-2,1,-1,1], id_ft=0)
plt.figure(figsize=(10,5))
#plt.plot(X_train[2,0], c='blue')
conv_points = set([0 + j*dils[i_kernel] for j in range(5)])
other_points = set(list(range(150)))
plt.scatter(list(other_points-conv_points),X_train[2,0][list(other_points-conv_points)], c='blue',s=25)
plt.scatter(list(conv_points),X_train[2,0][list(conv_points)], c='red',s=50)
plt.show()
plt.figure(figsize=(10,5))
conv = k._convolve_one_sample(X_train[2,0])
plt.scatter(range(1,conv.shape[0]),conv[1:],c='blue',s=25)
plt.scatter([0],conv[0:1],c='red',s=50)
plt.show()
plt.scatter(range(5),[1,-2,1,-1,1],c='green')
plt.show()



# In[]5
i_kernel = 7
print(dils[i_kernel])
class_locs = np.zeros((np.unique(y_train).shape[0],X_train.shape[2]))
fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15,5))
for c in np.unique(y_train):
    class_loc = np.sum(locs[np.where(y_train==c)[0],i_kernel],axis=0)
    class_loc = (class_loc - class_loc.mean())/class_loc.std()
    class_locs[c,:] = class_loc
    ax[c].plot(class_loc, c='green')
    ax[c].plot(X_train[np.where(y_train==c)[0][0],0],
               color='orange' if c==1 else 'blue')
    ax[c].set_title('LC of class {}'.format(c))
    

ax[2].plot(X_train[np.where(y_train==0)[0][0],0],c='blue',label='class 0')
ax[2].plot(X_train[np.where(y_train==1)[0][0],0],c='orange',label='class 1')
ax[2].plot(class_locs[0] - class_locs[1],c='green',label='LC')
ax[2].set_title('LC difference')
plt.legend()
plt.show()

"""
s = generate_strides_1D(class_locs[0] - class_locs[1], 9, dils[i_kernel]).sum(axis=1)
plt.plot(generate_strides_1D(X_train[np.random.choice(np.where(y_train==0)[0]),0], 9, dils[i_kernel]).sum(axis=1))
plt.plot(generate_strides_1D(X_train[np.random.choice(np.where(y_train==1)[0]),0], 9, dils[i_kernel]).sum(axis=1))
plt.plot(generate_strides_1D(class_locs[0] - class_locs[1], 9, dils[i_kernel]).sum(axis=1))
"""
# In[]:
i_kernel = 1
from itertools import combinations
for w in np.array([_ for _ in combinations(np.arange(9), 3)], dtype=np.int32):
    wt = np.zeros(9) + 1
    wt[w] = -2    
    k = Rocket_kernel(length=9, bias=biases[i_kernel], dilation=dils[i_kernel], padding=0, weights=wt, id_ft=0)
    ft = k.get_features(X_train)
    fig, ax = plt.subplots(ncols=3, sharex=False, sharey=False, figsize=(15,5))
    ax[0].plot(X_train[0,0],c='orange')
    ax[0].plot(X_train[2,0],c='blue')
    ax[0].set_title('Inputs')
    ax[1].plot(k._convolve_one_sample(X_train[0,0]),c='orange')
    ax[1].plot(k._convolve_one_sample(X_train[2,0]),c='blue')
    ax[1].set_title('Convolved Inputs')
    ax[2].scatter(ft[:,0],ft[:,1],facecolors='none',edgecolors=['blue' if c==0 else 'orange' for c in y_train],alpha=0.5)
    ax[2].scatter(ft[0,0],ft[0,1],c='orange')
    ax[2].scatter(ft[2,0],ft[2,1],c='blue')
    ax[2].set_xlabel('PPV')
    ax[2].set_ylabel('Max')
    ax[2].set_title('Rocket Features')
    plt.show()

    