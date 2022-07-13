# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:10:32 2022

@author: a694772
"""

import numpy as np
from numba import types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError
from numpy.random import seed as _seed, uniform as _uniform
from numba import njit, float64


@njit(fastmath = True, nogil = True, cache = True)
def arange(i):
	return np.arange(i)

@njit(fastmath = True, nogil = True, cache = True)
def arange2(a, b):
	return np.arange(a, b)

@njit(fastmath = True, nogil = True, cache = True)
def arange3(a, b, c):
	return np.arange(a, b, c)

@njit(fastmath = True, nogil = True, cache = True)
def _abs(v):
	return np.abs(v)

@njit(fastmath = True, nogil = True, cache = True)
def empty(shape, dtype=float64):
    return np.empty(shape, dtype=dtype)

@njit(fastmath = True, nogil = True, cache = True)
def zeros(shape, dtype=float64):
    return np.zeros(shape, dtype=dtype)

@njit(fastmath = True, nogil = True, cache = True)
def ones(shape, dtype=float64):
    return np.ones(shape, dtype=dtype)

@njit(nogil = True, cache = True)
def seed(v):
    return _seed(v)

@njit(fastmath = True, nogil = True, cache = True)
def uniform(v):
    return _uniform(0, v)

@njit(fastmath = True, nogil = True, cache = True)
def sum_axis(a, axis):
    return np.sum(a, axis=axis)

@njit(fastmath = True, nogil = True, cache = True)
def _sum(a):
    return np.sum(a)

@njit(fastmath = True, nogil = True, cache = True)
def cumsum(a):
    return np.cumsum(a)

@njit(fastmath = True, nogil = True, cache = True)
def power(b,v):
    return np.power(b,v)

@njit(fastmath = True, nogil = True, cache = True)
def sqrt(v):
    return np.sqrt(v)

@njit(fastmath = True, nogil = True, cache = True)
def floor_divide(a,b):
    return np.floor_divide(a,b)

@njit(fastmath = True, nogil = True, cache = True)
def mean_axis(a, axis):
    return np.sum(a, axis=axis)/a.shape[axis]

@njit(fastmath = True, nogil = True, cache = True)
def mean(a):
    return np.sum(a)/a.size

@njit(fastmath = True, nogil = True, cache = True)
def std_axis(a, axis):
    return np.std(a, axis=axis)

@njit(fastmath = True, nogil = True, cache = True)
def std(a):
    return np.std(a)

@njit(fastmath = True, nogil = True, cache = True)
def argmin(a):
    return np.argmin(a)

@njit(fastmath = True, nogil = True, cache = True)
def _min(a):
    return np.min(a)

@njit(fastmath = True, nogil = True, cache = True)
def _max(a):
    return np.max(a)

@njit(fastmath = True, nogil = True, cache = True)
def log2(v):
    return np.log2(v)

@njit(fastmath = True, nogil = True, cache = True)
def floor(v):
    return np.floor(v)

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
