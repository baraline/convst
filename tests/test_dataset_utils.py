# -*- coding: utf-8 -*-

from convst.utils.dataset_utils import (
    load_UCR_UEA_dataset_split, load_UCR_UEA_dataset, z_norm_3D
)
import numpy as np
import pytest


def init_numpy(dims):
    return np.random.random_sample(dims)

"""
@pytest.mark.parametrize("dims", [
    ((30, 15, 12)),
    ((2, 1, 18)),
])
"""

def test_z_norm_3D():
    X0 = np.zeros((10,1,10))
    X1 = np.ones((10,1,10))
    assert np.array_equal(z_norm_3D(X0), X0)
    assert np.array_equal(z_norm_3D(X1), X0)
    
    
@pytest.mark.parametrize("name", [
    ('GunPoint'), ('SmoothSubspace'),
])
def test_load_UCR_UEA_dataset_split(name):
    X_train, X_test, y_train, y_test, le = load_UCR_UEA_dataset_split(
        name=name, normalize=False
    )
    if name == 'GunPoint':
        assert X_train.shape == (50,1,150)
        assert X_test.shape == (150,1,150)
    elif name == 'SmoothSubspace':
        assert X_train.shape == (150,1,15)
        assert X_test.shape == (150,1,15)
        
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

@pytest.mark.parametrize("name", [
    ('GunPoint'), ('SmoothSubspace'),
])
def test_load_UCR_UEA_dataset(name):
    X, y = load_UCR_UEA_dataset(
        name=name, normalize=False
    )
    if name == 'GunPoint':
        assert X.shape == (200,1,150)
    elif name == 'SmoothSubspace':
        assert X.shape == (300,1,15)
    assert y.shape[0] == X.shape[0]