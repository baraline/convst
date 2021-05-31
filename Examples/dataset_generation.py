# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:15:29 2021

@author: A694772
"""
import numpy as np

from CST.base_transformers.minirocket import MiniRocket

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_validate

from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer

from matplotlib import pyplot as plt


def _init_dataset(n_samples, n_timestamps, n_classes):
    X = np.zeros((n_samples, 1, n_timestamps))
    y = np.zeros(n_samples)
    n_sample_per_class = n_samples//n_classes
    r = n_samples % n_classes
    for i in range(n_classes):
        y[i*n_sample_per_class:(i+1)*n_sample_per_class] = i
    for i in range(r):
        y[-i] = i
    return X, y.astype(int)


def make_same_timestamps_diff_values(n_samples=50, n_timestamps=100, n_locs=3,
                                     n_classes=3, scale_diff=1, noise_coef=0.25,
                                     shape_coef=0.5):
    X, y = _init_dataset(n_samples, n_timestamps, n_classes)
    base_data = np.random.rand(n_timestamps)
    locs = np.random.choice(range(n_timestamps), n_locs, replace=False)
    base_values = np.random.uniform(
        low=shape_coef, high=shape_coef*10, size=(n_locs))
    for i in range(n_samples):
        noise = np.random.normal(0, noise_coef, n_timestamps)
        X[i, 0] = base_data + noise
        X[i, 0, locs] += (base_values)*((1+y[i])*scale_diff)
    return X, y


def make_same_timestamps_diff_pattern(n_samples=50, n_timestamps=100, pattern_len=5,
                                      n_classes=3, noise_coef=0.25, shape_coef=0.5):
    X, y = _init_dataset(n_samples, n_timestamps, n_classes)
    base_data = np.random.rand(n_timestamps)
    loc = np.random.choice(range(n_timestamps-pattern_len))
    base_values = [np.random.uniform(
        low=shape_coef, high=shape_coef*10, size=(pattern_len)) for i in np.unique(y)]
    for i in range(n_samples):
        noise = np.random.normal(0, noise_coef, n_timestamps)
        X[i, 0] = base_data + noise
        X[i, 0, loc:loc+pattern_len] += base_values[y[i]]
    return X, y


def make_diff_timestamps_diff_pattern(n_samples=50, n_timestamps=100, pattern_len=5,
                                      n_classes=3, noise_coef=0.25, shape_coef=0.5):
    X, y = _init_dataset(n_samples, n_timestamps, n_classes)
    base_data = np.random.rand(n_timestamps)
    loc = np.random.choice(range(n_timestamps-pattern_len),
                           np.unique(y).shape[0], replace=False)
    base_values = [np.random.uniform(
        low=shape_coef, high=shape_coef*6, size=(pattern_len)) for i in np.unique(y)]
    for i in range(n_samples):
        noise = np.random.normal(0, noise_coef, n_timestamps)
        X[i, 0] = base_data + noise
        X[i, 0, loc[y[i]]:loc[y[i]]+pattern_len] += base_values[y[i]]
    return X, y


def make_diff_timestamps_same_pattern(n_samples=50, n_timestamps=100, pattern_len=5,
                                      n_classes=3, noise_coef=0.25, shape_coef=0.5):
    X, y = _init_dataset(n_samples, n_timestamps, n_classes)
    base_data = np.random.rand(n_timestamps)
    loc = np.random.choice(range(n_timestamps-pattern_len),
                           np.unique(y).shape[0], replace=False)
    base_values = np.random.uniform(
        low=shape_coef, high=shape_coef*6, size=(pattern_len))
    for i in range(n_samples):
        noise = np.random.normal(0, noise_coef, n_timestamps)
        #X[i,0] = base_data + noise
        X[i, 0, loc[y[i]]:loc[y[i]]+pattern_len] += base_values
    return X, y


def make_shift_different_pattern(n_samples=50, n_timestamps=100, pattern_len=20,
                                 n_classes=3, noise_coef=0.25, shape_coef=0.75, shift_coef=0.25):
    X, y = _init_dataset(n_samples, n_timestamps, n_classes)
    base_data = np.random.rand(n_timestamps)
    loc = np.random.choice(range(n_timestamps-pattern_len),
                           np.unique(y).shape[0], replace=False)
    base_values = [np.random.uniform(
        low=shape_coef, high=shape_coef*6, size=(pattern_len)) for i in np.unique(y)]
    for i in range(n_samples):
        noise = np.random.normal(0, noise_coef, n_timestamps)
        X[i, 0] = base_data + noise
        l = loc[y[i]]
        # TODO bug sometimes empty
        l += np.random.choice(range(int((n_timestamps -
                                         (pattern_len+l))*shift_coef)))
        X[i, 0, l:l+pattern_len] += base_values[y[i]]
    return X, y

# Meme Pattern qui commence a un point x, chaque class à une fréquence différente (20/40 par ex)


"""
[
make_same_timestamps_diff_values,
    Moyen, le nombre de bin joue un role mais ne fait pas tout, 
    le fait que ca sois sur la même loc doit jouer, 
    ok si la diff est assez importante
make_same_timestamps_diff_pattern,
    pas de probleme
make_diff_timestamps_diff_pattern,
     pas de probleme
make_diff_timestamps_same_pattern,
     la différence se fait uniquement sur la dilatation, dépend de si la 
     bonne dilatation est présente ou non pour différentié le probleme
make_shift_different_pattern
     pas de probleme
  ]
"""

for data_func in [
    make_same_timestamps_diff_values,
]:
    print(data_func.__name__)
    X, y = data_func(n_samples=100, n_timestamps=100, n_classes=5, noise_coef=0.1)
    color_dict = {0: 'green', 1: 'red', 2: 'blue', 3: 'orange', 4: 'brown'}
    for i in range(X.shape[0]):
        plt.plot(X[i, 0], c=color_dict[y[i]], alpha=0.1)
    plt.show()

    pipe_rkt = make_pipeline(ConvolutionalShapeletTransformer(),
                             RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True,
                                               class_weight='balanced'))
    
    cv = cross_validate(pipe_rkt, X, y, cv=3, scoring={
                        'f1': make_scorer(f1_score, average='macro')}, n_jobs=None)
    print("F1-Score for CST Ridge : {}".format(np.mean(cv['test_f1'])))
    
    pipe_rkt = make_pipeline(ConvolutionalShapeletTransformer(),
                             RandomForestClassifier(class_weight='balanced'))
    
    cv = cross_validate(pipe_rkt, X, y, cv=3, scoring={
                        'f1': make_scorer(f1_score, average='macro')}, n_jobs=None)
    print("F1-Score for CST RF : {}".format(np.mean(cv['test_f1'])))
    
    pipe_rkt = make_pipeline(MiniRocket(),
                             RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True,
                                               class_weight='balanced'))
    
    cv = cross_validate(pipe_rkt, X, y, cv=3, scoring={
                        'f1': make_scorer(f1_score, average='macro')}, n_jobs=None)
    print("F1-Score for ROCKET Ridge : {}".format(np.mean(cv['test_f1'])))
