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
ax[0].set_title("Class 0 of GunPoint (Gun)")
ax[1].set_title("Class 1 of GunPoint (NoGun)")
plt.show()

# In[]:
i_kernel = 0
print(dils[i_kernel])

wt = np.array([-2,1,1,1,1,-2,-2,1,1],dtype=np.float32)
fig, ax = plt.subplots(ncols=3,figsize=(15,5))

ax[0].plot(X_train[2,0],c='blue',label='class 0')
ax[0].plot(X_train[0,0],c='orange',label='class 1')
k1 = Rocket_kernel(length=9, bias=biases[i_kernel], dilation=dils[i_kernel], padding=0, weights=wt, id_ft=0)
ax[1].plot(k1._convolve_one_sample(X_train[2,0])>0,c='blue')
ax[1].plot(k1._convolve_one_sample(X_train[0,0])>0,c='orange')
x1 = k1.get_features(X_train)

k2 = Rocket_kernel(length=9, bias=biases[i_kernel]-0.25, dilation=dils[i_kernel], padding=0, weights=wt, id_ft=0)
ax[2].plot(k2._convolve_one_sample(X_train[2,0])>0,c='blue')
ax[2].plot(k2._convolve_one_sample(X_train[0,0])>0,c='orange')
x2 = k2.get_features(X_train)
plt.legend()
plt.show()

fig, ax = plt.subplots(ncols=2,figsize=(15,5))
ax[0].scatter(x1[:,0],x1[:,1],edgecolors=['blue' if c==0 else 'orange' for c in y_train])
ax[1].scatter(x2[:,0],x2[:,1],edgecolors=['blue' if c==0 else 'orange' for c in y_train])


# In[]5
i_kernel = 0
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


s = generate_strides_1D(class_locs[0] - class_locs[1], 9, dils[i_kernel]).sum(axis=1)
s = (s - s.mean()) / s.std()
plt.plot(generate_strides_1D(X_train[np.where(y_train==0)[0][0],0], 9, dils[i_kernel]).sum(axis=1))
plt.plot(generate_strides_1D(X_train[np.where(y_train==1)[0][0],0], 9, dils[i_kernel]).sum(axis=1))
plt.plot(s)

# In[]:

i_kernel = 0

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
    

# In[]:
from matplotlib import gridspec
from CST.base_transformers.shapelets import Convolutional_shapelet
fig, ax = plt.subplots(ncols=3, nrows=2,
                       sharex=False, sharey=False,
                       figsize=(15,7))

gs = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 1.5])
ax00 = plt.subplot(gs[0,0])
ax01 = plt.subplot(gs[0,1])
ax10 = plt.subplot(gs[1,0])
ax11 = plt.subplot(gs[1,1])
ax21 = plt.subplot(gs[1,2])

sv0 = X_train[2,0,25:50]
sv0 = (sv0 - sv0.mean())/sv0.std()
sv1 = X_train[0,0,35:60]
sv1 = (sv1 - sv1.mean())/sv1.std()
s0 = Convolutional_shapelet(values=sv0, dilation=1, padding=0, input_ft_id=0)
s1 = Convolutional_shapelet(values=sv1, dilation=1, padding=0, input_ft_id=0)

ax00.plot(sv0,c='red')
ax10.plot(sv1,c='black')

s0.plot_loc(X_train[0,0][25:90],ax=ax01,color='red',c_x='orange',x_alpha=0.5)
s1.plot_loc(X_train[0,0][25:90],ax=ax01,color='black',c_x='orange',x_alpha=0.5)
s0.plot_loc(X_train[2,0][25:90],ax=ax11,color='red',c_x='blue',x_alpha=0.5)
s1.plot_loc(X_train[2,0][25:90],ax=ax11,color='black',c_x='blue',x_alpha=0.5)

ax01.set_xticks(ticks=[0,20,40,60])
ax01.set_xticklabels(labels=[25,45,65,85])
ax11.set_xticks(ticks=[0,20,40,60])
ax11.set_xticklabels(labels=[25,45,65,85])

x = s0.transform(X_train)
y = s1.transform(X_train)

ax00.set_ylabel("S0")
ax00.set_title("Shapelets")
ax10.set_ylabel("S1")

ax01.set_title("Closest match on each class")
ax21.set_title("Shapelet Transform")
idx = list(set(list(range(X_train.shape[0]))) - {0,2})
ax21.scatter(x[0],y[0],c='blue')
ax21.scatter(x[2],y[2],c='orange')
ax21.scatter(x[idx], y[idx], alpha=0.75,facecolors='none',edgecolors=['orange' if c==1 else 'blue' for c in y_train[idx]])
ax21.set_xlabel('D( S0, X )')
ax21.set_ylabel('D( S1, X )')

plt.tight_layout()