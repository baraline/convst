# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:44:11 2022

@author: a694772
"""

from convst.transformers.RDST import *
import numpy as np
import pytest


def init_numpy(dims):
    return np.random.random_sample(dims)

def test_compute_shapelet_dist_vector():
    x = np.ones(10)
    x[[0,3,6,9]] = 2
    values = np.ones(3)
    values[0] = 2
    d_vect = compute_shapelet_dist_vector(x, values, 3, 1, 0.0)
    expected = np.array([0,2,2,0,2,2,0,2])
    assert np.array_equal(d_vect, expected)
    
def test_init_random_shapelet_params():
    values, lengths, dilations, threshold, normalize = _init_random_shapelet_params(
        200, np.array([7]), 100, 0.5
    )