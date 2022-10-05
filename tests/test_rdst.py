# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 20:26:34 2022

@author: antoi
"""
import pytest

from convst.classifiers import R_DST_Ridge
from convst.transformers import R_DST
from convst.utils.dataset_utils import load_sktime_dataset_split
from convst.utils.experiments_utils import cross_validate_UCR_UEA

#TODO test for each type of data

@pytest.mark.parametrize("name, expected", [
    ('GunPoint','BasicMotions'),
    ('univariate', 'multivariate')
])
def test_auto_type(name, expected):
    X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
        name=name
    )
    rdst = R_DST(n_shapelets=2).fit(X_train, y_train)
    assert rdst.transform_type == expected

@pytest.mark.parametrize("name, expected", [
    ('GunPoint','Wine','BasicMotions'),
    (0.98, 0.94, 0.94)
])
def test_performance(name, expected):
    X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
        name=name
    )
    rdst = R_DST_Ridge(n_shapelets=1).fit(X_train, y_train)
    rdst.score(X_test, y_test)
    assert rdst.transformer.shapelets_[1].shape[0] == 2
    acc = cross_validate_UCR_UEA(5,name).score(R_DST_Ridge(n_jobs=-1))
    acc = acc['accuracy'].mean()
    assert acc >= expected