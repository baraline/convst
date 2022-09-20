# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:10:32 2022

@author: a694772
"""

import numpy as np
from numba import types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
from numpy.random import uniform as _uniform
from numba import njit, prange


@njit(  
  cache=True
)
def arange(i):
    return np.arange(i)

arange(10)

@njit(
  cache=True
)
def arange2(a, b):
    return np.arange(a, b)

arange2(10, 2)

@njit(
  cache=True
)
def arange3(a, b, c):
    return np.arange(a, b, c)

arange3(10, 2, 2)

@njit(
  cache=True
)
def _abs(v):
    return abs(v)

_abs(-5)

@njit(
  cache=True
)
def empty(shape):
    return np.empty(shape)

empty(5)
empty((5,5))
empty((5,5,5))

@njit(
  cache=True
)
def zeros(shape):
    return np.zeros(shape)


zeros(5)
zeros((5,5))
zeros((5,5,5))

@njit(
  cache=True
)
def ones(shape):
    return np.ones(shape)


@njit(
  cache=True, fastmath=True
)
def uniform(v):
    return _uniform(0, v)

uniform(1)
uniform(1.5)

@njit(
  cache=True, fastmath=True
)
def sum_axis(a, axis):
    return np.sum(a, axis=axis)

sum_axis(np.ones((3,2),dtype=int),1)
sum_axis(np.ones((3,2),dtype=float),1)

@njit(
  cache=True, fastmath=True
)
def _sum(a):
    a = a.ravel()
    s = 0
    for i in prange(a.shape[0]):
        s+=a[i]
    return s

# TODO : numpy a bit faster for size <30, and >30000 

_sum(np.ones((3,2),dtype=int))
_sum(np.ones((3,2),dtype=float))
_sum(np.ones(3,dtype=int))
_sum(np.ones(3,dtype=float))


@njit(
  cache=True, fastmath=True
)
def cumsum(a):
    s = 0
    for i in prange(a.shape[0]):
        s += a[i]
        a[i] = s
    return a

cumsum(np.ones(3,dtype=int))
cumsum(np.ones(3,dtype=float))

@njit(
  cache=True, fastmath=True
)
def power(b,v):
    return np.power(b,v)

power(5, 2)
power(5, 2.1)
power(5.1, 2)
power(5.1, 2.1)

@njit(
  cache=True, fastmath=True
)
def sqrt(v):
    return np.sqrt(v)

sqrt(10)
sqrt(10.1)
sqrt(np.ones(3,dtype=int))
sqrt(np.ones(3,dtype=float))


@njit(
  cache=True, fastmath=True
)
def floor_divide(a,b):
    return np.floor_divide(a,b)

floor_divide(2,1)
floor_divide(2,1.1)
floor_divide(2.1,1)
floor_divide(2.1,1.1)
floor_divide(np.ones(3,dtype=int),np.ones(3,dtype=int))
floor_divide(np.ones(3,dtype=float),np.ones(3,dtype=float))
floor_divide(np.ones(3,dtype=int),np.ones(3,dtype=float))
floor_divide(np.ones(3,dtype=float),np.ones(3,dtype=int))

@njit(
  cache=True, fastmath=True
)
def mean_axis(a, axis):
    return np.sum(a, axis=axis)/a.shape[axis]

mean_axis(np.ones((3,2),dtype=int),1)
mean_axis(np.ones((3,2),dtype=float),1)

@njit(
  cache=True, fastmath=True
)
def mean(a):
    return _sum(a)/a.size

mean(np.ones(3,dtype=int))
mean(np.ones(3,dtype=float))

@njit(
  cache=True, fastmath=True
)
def std(a):
    return np.std(a)

std(np.ones(3,dtype=int))
std(np.ones(3,dtype=float))

@njit(
  cache=True, fastmath=True
)
def argmin(a):
    s = np.inf
    loc = 0
    for i in prange(a.shape[0]):
        if a[i] < s:
            s = a[i]
            loc = i
    return loc

argmin(np.ones(3,dtype=int))
argmin(np.ones(3,dtype=float))


@njit(
  cache=True, fastmath=True
)
def _min(a):
    s = np.inf
    for i in prange(a.shape[0]):
        if a[i] < s:
            s = a[i]
    return s

_min(np.ones(3,dtype=int))
_min(np.ones(3,dtype=float))


@njit(
  cache=True, fastmath=True
)
def _max(a):
    s = -np.inf
    for i in prange(a.shape[0]):
        if a[i] > s:
            s = a[i]
    return s

_max(np.ones(3,dtype=int))
_max(np.ones(3,dtype=float))

@njit(
  cache=True, fastmath=True
)
def log2(v):
    return np.log2(v)

log2(5)
log2(5.1)
log2(np.ones(3,dtype=int))
log2(np.ones(3,dtype=float))

@njit(
  cache=True, fastmath=True
)
def floor(v):
    return np.floor(v)

floor(5)
floor(5.1)
floor(np.ones(3,dtype=int))
floor(np.ones(3,dtype=float))

@overload(np.all)
def np_all(x, axis=None):

    # ndarray.all with axis arguments for 2D arrays.
    @register_jitable
    def _np_all_axis0(arr):
        out = np.logical_and(arr[0], arr[1])
        for v in iter(arr[2:]):
            for idx, v_2 in enumerate(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    @register_jitable
    def _np_all_flat(x):
        out = x.all()
        return out

    @register_jitable
    def _np_all_axis1(arr):
        out = np.logical_and(arr[:, 0], arr[:, 1])
        for idx, v in enumerate(arr[:, 2:]):
            for v_2 in iter(v):
                out[idx] = np.logical_and(v_2, out[idx])
        return out

    if isinstance(axis, types.Optional):
        axis = axis.type

    if not isinstance(axis, (types.Integer, types.NoneType)):
        raise TypingError("'axis' must be 0, 1, or None")

    if not isinstance(x, types.Array):
        raise TypingError("Only accepts NumPy ndarray")

    if not (1 <= x.ndim <= 2):
        raise TypingError("Only supports 1D or 2D NumPy ndarrays")

    if isinstance(axis, types.NoneType):

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 1:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

    elif x.ndim == 2:

        def _np_all_impl(x, axis=None):
            if axis == 0:
                return _np_all_axis0(x)
            else:
                return _np_all_axis1(x)

        return _np_all_impl

    else:

        def _np_all_impl(x, axis=None):
            return _np_all_flat(x)

        return _np_all_impl

@njit(cache=True)
def nb_unique(input_data, axis=0):
    """2D np.unique(a, return_index=True, return_counts=True)
    
    Parameters
    ----------
    input_data : 2D numeric array
    axis : int, optional
        axis along which to identify unique slices, by default 0
    Returns
    -------
    2D array
        unique rows (or columns) from the input array
    1D array of ints
        indices of unique rows (or columns) in input array
    1D array of ints
        number of instances of each unique row
    """

    # don't want to sort original data
    if axis == 1:
        data = input_data.T.copy()

    else:
        data = input_data.copy()

    # so we can remember the original indexes of each row
    orig_idx = np.array([i for i in range(data.shape[0])])

    # sort our data AND the original indexes
    for i in range(data.shape[1] - 1, -1, -1):
        sorter = data[:, i].argsort(kind="mergesort")

        # mergesort to keep associations
        data = data[sorter]
        orig_idx = orig_idx[sorter]
    # get original indexes
    idx = [0]

    if data.shape[1] > 1:
        bool_idx = ~np.all((data[:-1] == data[1:]), axis=1)
        additional_uniques = np.nonzero(bool_idx)[0] + 1

    else:
        additional_uniques = np.nonzero(~(data[:-1] == data[1:]))[0] + 1

    idx = np.append(idx, additional_uniques)
    # get counts for each unique row
    counts = np.append(idx[1:], data.shape[0])
    counts = counts - idx
    return data[idx], orig_idx[idx], counts
