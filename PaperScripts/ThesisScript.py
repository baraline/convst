# In[]:
    
from numba import njit, prange
from convst.utils.shapelets_utils import generate_strides_1D

@njit(fastmath=True, cache=True)
def apply_one_shapelet_one_sample(x, values):
    x = generate_strides_1D(x, values.shape[0], 1)
    n_candidates, length = x.shape
    candidates_index = np.arange(n_candidates)
    _min = 1e+100

    #For each step of the moving window in the shapelet distance
    for i in candidates_index:
        _dist = 0
        #For each value of the shapelet
        for j in prange(length):
            #Early abandon
            _dist += abs(x[i, j] - values[j])

        if _dist < _min:
            _min = _dist

    return _min


@njit(fastmath=True, cache=True)
def apply_one_shapelet_one_sample_ea_random(x, values):
    x = generate_strides_1D(x, values.shape[0], 1)
    n_candidates, length = x.shape
    candidates_index = np.arange(n_candidates)
    np.random.shuffle(candidates_index)
    _min = 1e+100

    #For each step of the moving window in the shapelet distance
    for i in candidates_index:
        _dist = 0
        #For each value of the shapelet
        for j in prange(length):
            #Early abandon
            _dist += abs(x[i, j] - values[j])
            if _dist > _min:
                break

        if _dist < _min:
            _min = _dist

    return _min


@njit(fastmath=True, cache=True)
def apply_one_shapelet_one_sample_ea(x, values):
    x = generate_strides_1D(x, values.shape[0], 1)
    n_candidates, length = x.shape
    candidates_index = np.arange(n_candidates)
    _min = 1e+100

    #For each step of the moving window in the shapelet distance
    for i in candidates_index:
        _dist = 0
        #For each value of the shapelet
        for j in prange(length):
            #Early abandon
            _dist += abs(x[i, j] - values[j])
            if _dist > _min:
                break

        if _dist < _min:
            _min = _dist

    return _min

@njit(fastmath=True, cache=True)
def apply_one_shapelet_ea_random(X, values):
    for i in prange(X.shape[0]):
        apply_one_shapelet_one_sample_ea_random(X[i], values)


@njit(fastmath=True, cache=True)
def apply_one_shapelet_ea(X, values):
    for i in prange(X.shape[0]):
        apply_one_shapelet_one_sample_ea(X[i], values)

@njit(fastmath=True, cache=True)
def apply_one_shapelet(X, values):
    for i in prange(X.shape[0]):
        apply_one_shapelet_one_sample(X[i], values)

import timeit
import pandas as pd
import numpy as np
from convst.utils.dataset_utils import load_sktime_dataset_split, return_all_dataset_names

data_names = return_all_dataset_names()
try:
    df = pd.read_csv('early_abandon_benchmark_2.csv',index_col=0)
except Exception:
    df = pd.DataFrame(0, index=data_names, columns=[])
    
X_train, y_train, _, _, _ = load_sktime_dataset_split('GunPoint')
apply_one_shapelet_ea_random(X_train[[0,1],0], X_train[0,0,0:10])
apply_one_shapelet_ea(X_train[[0,1],0], X_train[0,0,0:10])
apply_one_shapelet(X_train[[0,1],0], X_train[0,0,0:10])

for dataset_name in data_names:
    if 'mean_ea_time_0.01' in df.columns and not pd.isna(df.loc[dataset_name, 'mean_ea_time_0.01']):
        print('Skipping ' + dataset_name)
    else:
        print(dataset_name)
        try:
            X_train, y_train, _, _, _ = load_sktime_dataset_split(dataset_name)            
            n_samples, n_features, n_timestamps = X_train.shape
            df.loc[dataset_name, 'n_samples'] = n_samples
            df.loc[dataset_name, 'n_timestamps'] = n_timestamps
            for p in [0.01,0.025,0.05,0.1]:
                l = max(int(n_timestamps*p),3)
                d_len = n_timestamps-(l-1)
                res = []
                res_ea = []
                res_r = []
                i_shps = np.random.choice(n_samples * d_len, size=200)
                for i_shp in i_shps:
                    shp = X_train[i_shp//d_len,0,i_shp%d_len]
                    testcode = '''def test(): apply_one_shapelet_ea(X_train[:,0],shp)'''
                    res_ea.extend(timeit.repeat(stmt=testcode))
                    testcode = '''def test(): apply_one_shapelet(X_train[:,0],shp)'''
                    res.extend(timeit.repeat(stmt=testcode))
                    testcode = '''def test(): apply_one_shapelet_ea_random(X_train[:,0],shp)'''
                    res_r.extend(timeit.repeat(stmt=testcode))
                df.loc[dataset_name, 'mean_ea_time_'+str(p)] = np.mean(res_ea)
                df.loc[dataset_name, 'std_ea_time_'+str(p)] = np.std(res_ea)
                df.loc[dataset_name, 'mean_ear_time_'+str(p)] = np.mean(res_r)
                df.loc[dataset_name, 'std_ear_time_'+str(p)] = np.std(res_r)
                df.loc[dataset_name, 'mean_time_'+str(p)] = np.mean(res)
                df.loc[dataset_name, 'std_time_'+str(p)] = np.std(res)
                df.to_csv('early_abandon_benchmark_2.csv')
        except Exception as e:
            print(e)
# In[]:

import seaborn as sns
sns.set()
sns.set_context('talk')
from matplotlib import pyplot as plt
df = pd.read_csv('early_abandon_benchmark_2.csv',index_col=0)
df['n_data_points'] = df['n_samples'] * df['n_timestamps']
df = df.sort_values(by='n_data_points')
df = df.dropna(axis=0)
for p in [0.01,0.025,0.05,0.1]:
    dp = df[['mean_ea_time_'+str(p),'mean_ear_time_'+str(p),'mean_time_'+str(p)]]
    plt.figure(figsize=(10,10))
    dp = dp.cumsum()
    dp.plot()
    plt.show()
