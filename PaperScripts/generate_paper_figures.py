# -*- coding: utf-8 -*-


import matplotlib
from sklearn.ensemble import RandomForestClassifier
from CST.base_transformers.shapelets import Convolutional_shapelet
from matplotlib import gridspec
from CST.base_transformers.convolutional_kernels import Rocket_kernel
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from CST.utils.dataset_utils import load_sktime_dataset_split
from CST.utils.shapelets_utils import compute_distances, generate_strides_2D, generate_strides_1D
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")

# Load GunPoint Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'GunPoint', normalize=True)

CST = ConvolutionalShapeletTransformer()

locs, dils, biases, weights = CST._generate_inputs(X_train, y_train)
# In[]:

# GunPoint data Figure
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(15, 5))
for i in range(X_train.shape[0]):
    if y_train[i] == 1:
        ax[1].plot(X_train[i, 0], c='orange', alpha=0.2)
    else:
        ax[0].plot(X_train[i, 0], c='blue', alpha=0.2)
ax[0].set_title("Class 0 of GunPoint (Gun)")
ax[1].set_title("Class 1 of GunPoint (NoGun)")
plt.show()

# In[]:

i_kernel = 0

k = Rocket_kernel(length=9, bias=biases[i_kernel], dilation=dils[i_kernel],
                  padding=0, weights=weights[i_kernel], id_ft=0)
ft = k.get_features(X_train)
fig, ax = plt.subplots(ncols=3, sharex=False, sharey=False, figsize=(15, 5))
ax[0].plot(X_train[0, 0], c='orange')
ax[0].plot(X_train[2, 0], c='blue')
ax[0].set_title('Inputs')
ax[1].plot(k._convolve_one_sample(X_train[0, 0]), c='orange')
ax[1].plot(k._convolve_one_sample(X_train[2, 0]), c='blue')
ax[1].set_title('Convolved Inputs')
ax[2].scatter(ft[:, 0], ft[:, 1], facecolors='none', edgecolors=[
              'blue' if c == 0 else 'orange' for c in y_train], alpha=0.5)
ax[2].scatter(ft[0, 0], ft[0, 1], c='orange')
ax[2].scatter(ft[2, 0], ft[2, 1], c='blue')
ax[2].set_xlabel('PPV')
ax[2].set_ylabel('Max')
ax[2].set_title('Rocket Features')
plt.show()

# In[]:
fig, ax = plt.subplots(ncols=3, nrows=2,
                       sharex=False, sharey=False,
                       figsize=(15, 7))

gs = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 1.5])
ax00 = plt.subplot(gs[0, 0])
ax01 = plt.subplot(gs[0, 1])
ax10 = plt.subplot(gs[1, 0])
ax11 = plt.subplot(gs[1, 1])
ax21 = plt.subplot(gs[1, 2])

sv0 = X_train[2, 0, 25:50]
sv0 = (sv0 - sv0.mean())/sv0.std()
sv1 = X_train[0, 0, 35:60]
sv1 = (sv1 - sv1.mean())/sv1.std()
s0 = Convolutional_shapelet(values=sv0, dilation=1, padding=0, input_ft_id=0)
s1 = Convolutional_shapelet(values=sv1, dilation=1, padding=0, input_ft_id=0)

ax00.plot(sv0, c='red')
ax10.plot(sv1, c='black')

s0.plot_loc(X_train[0, 0][25:90], ax=ax01,
            color='red', c_x='orange', x_alpha=0.5)
s1.plot_loc(X_train[0, 0][25:90], ax=ax01,
            color='black', c_x='orange', x_alpha=0.5)
s0.plot_loc(X_train[2, 0][25:90], ax=ax11,
            color='red', c_x='blue', x_alpha=0.5)
s1.plot_loc(X_train[2, 0][25:90], ax=ax11,
            color='black', c_x='blue', x_alpha=0.5)

ax01.set_xticks(ticks=[0, 20, 40, 60])
ax01.set_xticklabels(labels=[25, 45, 65, 85])
ax11.set_xticks(ticks=[0, 20, 40, 60])
ax11.set_xticklabels(labels=[25, 45, 65, 85])

x = s0.transform(X_train)
y = s1.transform(X_train)

ax00.set_ylabel("S0")
ax00.set_title("Shapelets")
ax10.set_ylabel("S1")

ax01.set_title("Closest match on each class")
ax21.set_title("Shapelet Transform")
idx = list(set(list(range(X_train.shape[0]))) - {0, 2})
ax21.scatter(x[0], y[0], c='blue')
ax21.scatter(x[2], y[2], c='orange')
ax21.scatter(x[idx], y[idx], alpha=0.75, facecolors='none', edgecolors=[
             'orange' if c == 1 else 'blue' for c in y_train[idx]])
ax21.set_xlabel('D( S0, X )')
ax21.set_ylabel('D( S1, X )')

plt.tight_layout()
# In[]

i_kernel = 0

k = Rocket_kernel(length=9, bias=biases[i_kernel], dilation=dils[i_kernel],
                  padding=0, weights=weights[i_kernel], id_ft=0)

fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(18, 5))

Lp = generate_strides_2D(locs[:, i_kernel, :], 9, dils[i_kernel]).sum(axis=-1)
LC = np.zeros((np.unique(y_train).shape[0], Lp.shape[1]))
for c in np.unique(y_train):
    LC[c] = np.sum(Lp[np.where(y_train == c)[0]], axis=0)
    LC[c] = (LC[c] - LC[c].mean())/LC[c].std()
    ax[c].plot(LC[c], c='green', label='LC')
    x = k._convolve_one_sample(X_train[np.where(y_train == c)[0][0], 0])
    x = (x-x.mean())/x.std()
    ax[c].plot(x, color='orange' if c == 1 else 'blue',
               alpha=0.5, label='class {}'.format(c))
    ax[c].set_title('LC of class {}'.format(c))
    ax[c].legend(fontsize=15, loc='lower right')

x0 = k._convolve_one_sample(X_train[np.where(y_train == 0)[0][0], 0])
x0 = (x0-x0.mean())/x0.std()
x1 = k._convolve_one_sample(X_train[np.where(y_train == 1)[0][0], 0])
x1 = (x1-x1.mean())/x1.std()
ax[2].plot(x0, c='blue', label='class 0', alpha=0.5)
ax[2].plot(x1, c='orange', label='class 1', alpha=0.5)
D = LC[0] - LC[1]
ax[2].plot(D, c='green', label='D')
ax[2].axhline(np.percentile(D, 80), c='red', linestyle='--')

id_w = np.where(D >= np.percentile(D, 80))[0]
for i_r in CST._get_regions(id_w):
    ax[2].fill_between(i_r, D[i_r], np.percentile(
        D, 80), color='red', alpha=0.75)
ax[2].set_title('LC difference')

ax[3].plot(X_train[np.where(y_train == 0)[0][0], 0], c='blue', label='class 0')
ax[3].plot(X_train[np.where(y_train == 1)[0][0], 0],
           c='orange', label='class 1')
ax[3].legend(fontsize=15, loc='lower right')
region = [[i + j*dils[i_kernel] for j in range(9)] for i in id_w]
for r in region:
    ax[3].vlines(r, ymin=-1, ymax=0.5, color='red', alpha=0.1)
ax[3].set_title('View in input space')
plt.show()
# In[]
CST.fit(X_train, y_train)
d_X_train = CST.transform(X_train)
d_X_test = CST.transform(X_test)
rf = RandomForestClassifier(n_estimators=400).fit(d_X_train, y_train)
pred = rf.predict(d_X_test)

# In[]
x_test = X_test[0, 0]
y_x = y_test[0]
pred_x = pred[0]

x_coef = np.zeros((np.unique(y_test).shape[0], x_test.shape[0]))
n_coef = np.zeros((np.unique(y_test).shape[0], x_test.shape[0]))
for i_class in np.unique(y_test):
    for i, i_grp in enumerate(CST.shapelets_params.keys()):
        dilation, _ = CST.shapelets_params[i_grp]
        x_strides = generate_strides_1D(x_test, 9, dilation)
        i_shp = np.where(CST.shapelets_class[i] == i_class)[0]
        for j in range(i_shp.shape[0]):
            d = np.abs(
                x_strides - CST.shapelets_values[i][i_shp[j]]).sum(axis=1)
            dists = d.min()
            loc = d.argmin()
            x_coef[i_class, loc] += dists
            n_coef[i_class, loc] += 1

i = np.where(x_coef > 0)
x_coef[i] = x_coef[i] / n_coef[i]
norm = matplotlib.colors.Normalize(vmin=x_coef.min(), vmax=x_coef.max())
cmap = matplotlib.cm.ScalarMappable(
    norm=norm, cmap=matplotlib.cm.get_cmap('jet'))
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(15, 5))


ax[0].scatter(range(x_test.shape[0]), x_test, c=cmap.to_rgba(x_coef[0]), s=40)
ax[0].set_title('Class 0')
ax[1].scatter(range(x_test.shape[0]), x_test, c=cmap.to_rgba(x_coef[1]), s=40)
ax[1].set_title('Class 1')
fig.colorbar(cmap,
             ax=ax[1], orientation='vertical')
