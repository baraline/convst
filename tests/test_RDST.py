# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:44:11 2022

@author: a694772
"""
import numpy as np
from convst.transformers import R_DST, MR_DST, GR_DST
from convst.utils.dataset_utils import load_sktime_dataset_split
import pytest

#TODO : test individual functions

def test_all_variations():
    X_train, X_test, y_train, y_test, le = load_sktime_dataset_split('GunPoint')

    R_DST(n_shapelets=5).fit_transform(X_train, y_train)
    X_train2 = np.concatenate((X_train, X_train),axis=1)
    X_train3 = list()
    for i in range(X_train2.shape[0]):
        X_train3.append(X_train2[i,:,:np.random.choice(range(100,150))])

    MR_DST(n_shapelets=5).fit_transform(X_train2, y_train)
    GR_DST(n_shapelets=5).fit_transform(X_train3, y_train)
