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

import logging

LOGGER = logging.getLogger(__name__)

@pytest.mark.parametrize("name, expected", [
    ('GunPoint','univariate'),
    ('BasicMotions','multivariate'),
    ('PLAID','univariate_variable'),
    ('AsphaltObstaclesCoordinates','multivariate_variable')
])
def test_auto_type(name, expected):
    X_train, X_test, y_train, y_test, min_len = load_sktime_dataset_split(
        name=name
    )
    rdst = R_DST(n_shapelets=2, min_len=min_len).fit(X_train, y_train)
    assert rdst.transform_type == expected


@pytest.mark.parametrize("name, lengths", [
    ('GunPoint',[11]),
    ('GunPoint',[0.05]),
    ('GunPoint',[11,15,19]),
    ('GunPoint',[0.05,0.08,0.1]),
    ('BasicMotions',[11]),
    ('BasicMotions',[0.05]),
    ('BasicMotions',[11,15,19]),
    ('BasicMotions',[0.05,0.08,0.1]),
    ('PLAID',[11]),
    ('PLAID',[0.05]),
    ('PLAID',[11,15,19]),
    ('PLAID',[0.05,0.08,0.1]),
    ('AsphaltObstaclesCoordinates',[11]),
    ('AsphaltObstaclesCoordinates',[0.05]),
    ('AsphaltObstaclesCoordinates',[11,15,19]),
    ('AsphaltObstaclesCoordinates',[0.05,0.08,0.1])
])
def test_mutliple_lengths(name, lengths):
    X_train, X_test, y_train, y_test, min_len = load_sktime_dataset_split(
        name=name
    )
    try:
        R_DST(n_shapelets=1000, shapelet_lengths=lengths, min_len=min_len).fit(X_train, y_train)
    except Exception as e:
        LOGGER.info('A data format test failed on {} due to the following exception : {}'.format(
             name, e  
        ))
        assert False 
    assert True

# Lower than actual best accuracy to account for possible deviation due to random sampling
@pytest.mark.parametrize("name, expected", [
    ('GunPoint',0.98),
    ('Wine',0.94),
    ('BasicMotions',0.94),
    ('PLAID',0.895),
    ('AsphaltObstaclesCoordinates',0.795)
])
def test_performance(name, expected):
    X_train, X_test, y_train, y_test, min_len = load_sktime_dataset_split(
        name=name
    )
    rdst = R_DST_Ridge(n_shapelets=1,min_len=min_len).fit(X_train, y_train)
    rdst.score(X_test, y_test)
    assert rdst.transformer.shapelets_[1].shape[0] == 1
    acc = cross_validate_UCR_UEA(5,name).score(R_DST_Ridge(n_jobs=-1,min_len=min_len))
    acc = acc['accuracy'].mean()
    LOGGER.info('{} Dataset -> Accuracy {}, Expected >= {}'.format(
         name, acc, expected   
    ))
    assert acc >= expected