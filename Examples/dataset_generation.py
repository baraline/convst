# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:15:29 2021

@author: A694772
"""
import numpy as np
from matplotlib import pyplot as plt

"""
n_samples : int, default=100
    The number of samples.

n_features : int, default=20
    The total number of features. These comprise n_informative informative features, n_redundant redundant features, n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at random.

n_informative : int, default=2
    The number of informative features. Each class is composed of a number of gaussian clusters each located around the vertices of a hypercube in a subspace of dimension n_informative. For each cluster, informative features are drawn independently from N(0, 1) and then randomly linearly combined within each cluster in order to add covariance. The clusters are then placed on the vertices of the hypercube.

n_redundant : int, default=2
    The number of redundant features. These features are generated as random linear combinations of the informative features.

n_repeated : int, default=0
    The number of duplicated features, drawn randomly from the informative and the redundant features.

n_classes : int, default=2
    The number of classes (or labels) of the classification problem.

n_clusters_per_class : int, default=2
    The number of clusters per class.

weights : array-like of shape (n_classes,) or (n_classes - 1,), default=None
    The proportions of samples assigned to each class. If None, then classes are balanced. Note that if len(weights) == n_classes - 1, then the last class weight is automatically inferred. More than n_samples samples may be returned if the sum of weights exceeds 1. Note that the actual class proportions will not exactly match weights when flip_y isn’t 0.

flip_y : float, default=0.01
    The fraction of samples whose class is assigned randomly. Larger values introduce noise in the labels and make the classification task harder. Note that the default setting flip_y > 0 might lead to less than n_classes in y in some cases.

class_sep : float, default=1.0
    The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the classification task easier.

hypercube : bool, default=True
    If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of a random polytope.

shift : float, ndarray of shape (n_features,) or None, default=0.0
    Shift features by the specified value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].

scale : float, ndarray of shape (n_features,) or None, default=1.0
    Multiply features by the specified value. If None, then features are scaled by a random value drawn in [1, 100]. Note that scaling happens after shifting.

shuffle : bool, default=True
    Shuffle the samples and the features.
"""

def _init_dataset(n_samples, n_timestamps, n_classes):
    X = np.zeros((n_samples,1,n_timestamps))
    y = np.zeros(n_samples)
    n_sample_per_class = n_samples//n_classes
    r = n_samples%n_classes
    for i in range(n_classes):
        y[i*n_sample_per_class:(i+1)*n_sample_per_class] = i
    for i in range(r):
        y[-i] = i
    return X, y


#To add :
#Diff timestamps, same value
#Diff timestamps, diff value
#Shift
def make_same_timestamps_diff_values(n_samples=100, n_timestamps=50, n_locs=3,
                                     n_classes=3, scale_diff=1., noise_coef=0.25):
    X, y = _init_dataset(n_samples, n_timestamps, n_classes)
    base_data = np.random.rand(n_timestamps)
    locs = np.random.choice(range(n_timestamps), n_locs, replace=False)
    base_values = np.random.uniform(low=noise_coef*2, high=noise_coef*6, size=(n_locs))
    for i in range(n_samples):
        noise = np.random.normal(0,noise_coef,n_timestamps)
        X[i,0] = base_data + noise    
        X[i,0,locs] += (base_values)*((1+y[i])*scale_diff)
    print(locs)
    return X, y

X, y = make_same_timestamps_diff_values()
print(y)

color_dict = {0:'green',1:'red',2:'blue'}
for i in range(X.shape[0]):
    plt.plot(X[i,0], c=color_dict[y[i]],alpha=0.1)
plt.show()

