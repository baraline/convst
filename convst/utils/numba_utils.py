# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:10:32 2022

@author: a694772
"""

from numpy import arange as _arange, abs as __abs, empty as _empty, \
    zeros as _zeros, ones as _ones, log2 as _log2, floor as _floor,\
    sum as __sum, mean as _mean, max as __max, min as __min, std as _std,\
    floor_divide as _floor_divide, sqrt as _sqrt, power as _power,\
    argmin as _argmin, cumsum as _cumsum
    
from numpy.random import seed as _seed, uniform as _uniform
from numba import njit, float64

@njit(fastmath = True, nogil = True, cache = True)
def arange(i):
	return _arange(i)

@njit(fastmath = True, nogil = True, cache = True)
def arange2(a, b):
	return _arange(a, b)

@njit(fastmath = True, nogil = True, cache = True)
def arange3(a, b, c):
	return _arange(a, b, c)

@njit(fastmath = True, nogil = True, cache = True)
def _abs(v):
	return __abs(v)

@njit(fastmath = True, nogil = True, cache = True)
def empty(shape, dtype=float64):
    return _empty(shape, dtype=dtype)

@njit(fastmath = True, nogil = True, cache = True)
def zeros(shape, dtype=float64):
    return _zeros(shape, dtype=dtype)

@njit(fastmath = True, nogil = True, cache = True)
def ones(shape, dtype=float64):
    return _ones(shape, dtype=dtype)

@njit(nogil = True, cache = True)
def seed(v):
    return _seed(v)

@njit(fastmath = True, nogil = True, cache = True)
def uniform(v):
    return _uniform(0, v)

@njit(fastmath = True, nogil = True, cache = True)
def sum_array(a, axis):
    return __sum(a, axis=axis)

@njit(fastmath = True, nogil = True, cache = True)
def _sum(a):
    return __sum(a)

@njit(fastmath = True, nogil = True, cache = True)
def cumsum(a):
    return _cumsum(a)

@njit(fastmath = True, nogil = True, cache = True)
def power(b,v):
    return _power(b,v)

@njit(fastmath = True, nogil = True, cache = True)
def sqrt(v):
    return _sqrt(v)

@njit(fastmath = True, nogil = True, cache = True)
def floor_divide(a,b):
    return _floor_divide(a,b)

@njit(fastmath = True, nogil = True, cache = True)
def mean_array(a, axis):
    return __sum(a, axis=axis)/a.shape[axis]

@njit(fastmath = True, nogil = True, cache = True)
def mean(a):
    return __sum(a)/a.size

@njit(fastmath = True, nogil = True, cache = True)
def std_array(a, axis):
    return _std(a, axis=axis)

@njit(fastmath = True, nogil = True, cache = True)
def std(a):
    return _std(a)

@njit(fastmath = True, nogil = True, cache = True)
def argmin(a):
    return _argmin(a)

@njit(fastmath = True, nogil = True, cache = True)
def _min(a):
    return __min(a)

@njit(fastmath = True, nogil = True, cache = True)
def _max(a):
    return __max(a)

@njit(fastmath = True, nogil = True, cache = True)
def log2(v):
    return _log2(v)

@njit(fastmath = True, nogil = True, cache = True)
def floor(v):
    return _floor(v)

