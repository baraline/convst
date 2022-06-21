# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:52:25 2022

@author: a694772
"""

import numpy as np
from numba import njit, prange
from numba import types
from numba.extending import overload, register_jitable
from numba.core.errors import TypingError

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

@njit()
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


@njit()
def shapelets_to_mcb_quantile(S, n_symbols):
    S_new = np.empty(S.shape)
    n_shp, length = S.shape
    symbols = np.arange(n_symbols)
    step = 100//n_symbols
    for i in prange(length):
        min_bound = S[:,i].min() -0.1
        for ix, perc in enumerate(range(step,100,step)):
            max_bound = np.percentile(S[:,i], perc)
            S_new[(min_bound<S[:,i]) & (S[:,i]<=max_bound), i] = symbols[ix]
            min_bound = max_bound
        max_bound = S[:,i].max()
        S_new[(min_bound<S[:,i]) & (S[:,i]<=max_bound), i] = symbols[-1]
    return S_new
        
@njit()
def shapelets_to_mcb_uniform(S, n_symbols):
    S_new = np.empty(S.shape)
    n_shp, length = S.shape
    symbols = np.arange(n_symbols)
    for i in prange(length):
        min_bound = S[:,i].min() - 1e-8
        max_bound = S[:,i].max() + 1e-8
        step = abs(max_bound-min_bound)/n_symbols
        current_step = min_bound
        for ix in range(n_symbols):
            S_new[(current_step<S[:,i]) & (S[:,i]<=current_step+step), i] = symbols[ix]
            current_step += step
    return S_new

@njit()
def shapelets_to_sax(S, n_symbols):
    S_new = np.empty(S.shape[0]*S.shape[1])
    n_shp, length = S.shape
    symbols = np.arange(n_symbols)
    step = 100//n_symbols
    min_bound = S.min() -0.1
    for ix, perc in enumerate(range(step,100,step)):
        max_bound = np.percentile(S, perc)
        mask = ((min_bound<S) & (S<=max_bound)).reshape(-1)
        S_new[mask] = symbols[ix]
        min_bound = max_bound
    max_bound = S.max()
    mask = ((min_bound<S) & (S<=max_bound)).reshape(-1)
    S_new[mask] = symbols[-1]
    return S_new.reshape(S.shape)

@njit()
def count_occ(S, y):
    all_u, _, _ = nb_unique(S, axis=0)
    n_classes = y.max()+1
    count_per_y = np.zeros((n_classes, all_u.shape[0]))
    for i in range(n_classes):
        u_shp, _, count = nb_unique(S[y==i],axis=0)
        for j in range(u_shp.shape[0]):
            mask = (all_u == u_shp[j])
            count_per_y[i][np.all(mask, axis=1)] += count[j]
    return all_u, count_per_y

@njit()
def unique_per_X(S, y, ix):
    S_new = np.empty(S.shape)
    ix_new = np.empty(S.shape[0],dtype=np.int64)
    y_new = np.empty(S.shape[0],dtype=np.int64)
    j = 0
    u_ix = np.unique(ix)
    
    for i in range(u_ix.shape[0]):
        u,_,_ = nb_unique(S[ix==u_ix[i]])
        S_new[j:j+u.shape[0]] = u
        ix_new[j:j+u.shape[0]] = u_ix[i]
        y_new[j:j+u.shape[0]] = y[ix==u_ix[i]][0]
        j+= u.shape[0]
    return S_new[:j], y_new[:j], ix_new[:j]

@njit(fastmath=True, cache=True)
def znorm(S):
    S_new = np.zeros(S.shape)
    for i in prange(S.shape[0]):
        S_new[i] = (S[i] - S[i].mean()) / (S[i].std() + 1e-8)
    return S_new



# In[]:
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_context()
from convst.utils.dataset_utils import load_sktime_dataset_split
from convst.utils.shapelets_utils import generate_strides_2D
X,_,y,_,_ = load_sktime_dataset_split('GunPoint')
n_strides = X.shape[2] - (8*3)
S = generate_strides_2D(X[:,0,:], 9, 1).reshape(-1,9)
S = znorm(S)

idx_strides = np.zeros(S.shape[0], dtype=np.int64)
y_strides = np.zeros(S.shape[0], dtype=np.int64)
for i in range(y.shape[0]):
    y_strides[i*n_strides:(i+1)*n_strides] = y[i]
    idx_strides[i*n_strides:(i+1)*n_strides] = i

S_mcb = shapelets_to_sax(S, 5)
print("{}".format(S_mcb.shape))
S_mcb, y_strides, idx_strides = unique_per_X(S_mcb, y_strides, idx_strides)
print("{}".format(S_mcb.shape))


u,c = count_occ(S_mcb, y_strides)
print("{} -> {}".format(S_mcb.shape, u.shape))

m = ((1+c[0])/(1+c[0].sum())) / ((1+c[1])/(1+c[1].sum()))

lift_c0 = c[0]/(c.sum(axis=0)*c[0].sum())
lift_c1 = c[1]/(c.sum(axis=0)*c[1].sum())

conf_c0 = c[0]/c.sum(axis=0)
conf_c1 = c[1]/c.sum(axis=0)
diff = ((c[0]+1)/(c[1]+1))
id_s = m.argsort()[::-1][0:5]
for ix in id_s:
    plt.figure(figsize=(15,5))
    print("Shp : {}".format(u[ix]))
    print("Counts : {}".format(c[:,ix]))
    plt.plot(u[ix],alpha=0.95)
    plt.show()

# In[]:

from convst.utils.shapelets_utils import generate_strides_1D
n_ts = 150+11
n_sp = 50
l = 11
d = 1
t=2
n_conv = n_ts-(l-1)*d
p = np.zeros(n_ts)
to_zeros = np.zeros(n_conv,dtype=np.bool_)
for i in range(n_conv):
    if i<d or i>=n_conv-d or (i//d)%(t+1)==0:
        for j in range(l):
            p[i+(j*d)] += 1
    else:
        to_zeros[i] = True
        
p = (1/p)
p = generate_strides_1D(p, l, d).sum(axis=1)
p[to_zeros] = 0
p = p/p.sum()
cm = p.cumsum()
r = np.random.rand()
fig,ax = plt.subplots(ncols=2,figsize=(15,5))
ax[0].plot(cm-r)
ax[1].plot(1/(cm-r))
#argmax
plt.show()

# In[]:

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from convst.utils.dataset_utils import load_sktime_dataset_split

X,X1,y,y1,_ = load_sktime_dataset_split('EOGVerticalSignal')
# In[]:
    

pipeline_RDST = make_pipeline(R_DST(n_shapelets=10000), 
                                 StandardScaler(with_mean=False),
                                 RidgeClassifierCV(np.logspace(-6,6,20)))
pipeline_RDST.fit(X,y)
print(pipeline_RDST.score(X1,y1))

