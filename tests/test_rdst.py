# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 20:26:34 2022

@author: antoi
"""
import pytest

from convst.classifiers import R_DST_Ridge
from convst.utils.dataset_utils import load_sktime_dataset_split

#TODO test for each type of data

@pytest.mark.parametrize("name", [
    ('GunPoint'),
])
def test_load_sktime_dataset_split(name):
    X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
        name=name
    )
    rdst = R_DST_Ridge(n_shapelets=2).fit(X_train, y_train)
    rdst.score(X_test, y_test)
    assert rdst.transformer.shapelets_[1].shape[0] == 2