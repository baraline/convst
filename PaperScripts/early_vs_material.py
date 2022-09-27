# In[]:
    
from numba import njit, prange
from convst.utils.shapelets_utils import generate_strides_1D
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from convst.utils.dataset_utils import load_sktime_dataset_split, return_all_dataset_names
import seaborn as sns

# Re run the script by commenting njits to get py results

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


def time_func(f, x, shp):
    t0 = timer()    
    f(x, shp)
    t1 = timer()
    return t1-t0


df_name = 'early_abandon_benchmark.csv'
data_names = return_all_dataset_names()
try:
    df = pd.read_csv(df_name,index_col=0)
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
                i_shps = np.random.choice(n_samples * d_len, size=100)
                for i_shp in i_shps:
                    shp = X_train[i_shp//d_len,0,i_shp%d_len:i_shp%d_len+l]
                    for j in range(10):
                        res_ea.append(time_func(apply_one_shapelet_ea,X_train[:,0],shp))
                        res.append(time_func(apply_one_shapelet,X_train[:,0],shp))
                        res_r.append(time_func(apply_one_shapelet_ea_random,X_train[:,0],shp))
                    
                df.loc[dataset_name, 'mean_ea_time_'+str(p)] = np.mean(res_ea)
                df.loc[dataset_name, 'std_ea_time_'+str(p)] = np.std(res_ea)
                df.loc[dataset_name, 'mean_ear_time_'+str(p)] = np.mean(res_r)
                df.loc[dataset_name, 'std_ear_time_'+str(p)] = np.std(res_r)
                df.loc[dataset_name, 'mean_time_'+str(p)] = np.mean(res)
                df.loc[dataset_name, 'std_time_'+str(p)] = np.std(res)
                df.to_csv(df_name)
        except Exception as e:
            print(e)
# In[]:


sns.set()
sns.set_context('talk')
from matplotlib import pyplot as plt
df = pd.read_csv('early_abandon_benchmark.csv',index_col=0)
df2 = pd.read_csv('early_abandon_benchmark_py.csv',index_col=0)
df_sum = pd.DataFrame()
trad = {
    'ea_time_':'early abandon',
    'ear_time_':'early abandon + random order',
    'time_':'no speed-up',   
}
for p in [0.01,0.025,0.05,0.1]:
    for r in ['ea_time_','ear_time_','time_']:
        _s_mean = 0
        _s_min = 0
        _s_max = 0
        for dataset in df.index.values:
            _s_base = df.loc[dataset,'mean_'+r+str(p)]
            _s_std = df.loc[dataset,'std_'+r+str(p)]
            _s_mean += _s_base
            _s_min += _s_base - _s_std
            _s_max += _s_base + _s_std
        
        df_sum.loc[p,trad[r]+' Numba'] = _s_mean
        #df_sum.loc[p,trad[r]+' Numba (-std)'] = _s_min
        #df_sum.loc[p,trad[r]+' Numba (+std)'] = _s_max
        
        for dataset in df.index.values:
            _s_base = df2.loc[dataset,'mean_'+r+str(p)]
            _s_std = df2.loc[dataset,'std_'+r+str(p)]
            _s_mean += _s_base
            _s_min += _s_base - _s_std
            _s_max += _s_base + _s_std
        
        df_sum.loc[p,trad[r]+' Python'] = _s_mean
        #df_sum.loc[p,trad[r]+' Python (-std)'] = _s_min
        #df_sum.loc[p,trad[r]+' Python (+std)'] = _s_max

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), sharey=True)
r_names = ['early abandon Numba','early abandon + random order Numba','no speed-up Numba', 'early abandon Python','early abandon + random order Python','no speed-up Python']
ax[0,0].set_xscale('log')
ax[0,0].set_xticks(ticks=[0.06, 0.6, 6, 60])
ax[0,0].set_xticklabels(labels=['60ms','600ms', '6s', '1min'])
ax[0,0].set_xticklabels(labels=['60ms','600ms', '6s', '1min'])

ax[0,1].set_xscale('log')
ax[0,1].set_xticks(ticks=[0.06, 0.6, 6, 60])
ax[0,1].set_xticklabels(labels=['60ms','600ms', '6s', '1min'])
ax[0,1].set_xticklabels(labels=['60ms','600ms', '6s', '1min'])

ax[1,0].set_xscale('log')
ax[1,0].set_xticks(ticks=[0.06, 0.6, 6, 60])
ax[1,0].set_xticklabels(labels=['60ms','600ms', '6s', '1min'])
ax[1,0].set_xticklabels(labels=['60ms','600ms', '6s', '1min'])

ax[1,1].set_xscale('log')
ax[1,1].set_xticks(ticks=[0.06, 0.6, 6, 60])
ax[1,1].set_xticklabels(labels=['60ms','600ms', '6s', '1min'])
ax[1,1].set_xticklabels(labels=['60ms','600ms', '6s', '1min'])

ax[1,0].set_yticks(range(len(r_names)))
ax[1,0].set_yticklabels(labels=r_names)
ax[0,0].set_yticks(range(len(r_names)))
ax[0,0].set_yticklabels(labels=r_names)

ax[0,0].set_title('Shapelet length = 0.01')
ax[0,1].set_title('Shapelet length = 0.025')
ax[1,0].set_title('Shapelet length = 0.05')
ax[1,1].set_title('Shapelet length = 0.1')

ranks = df_sum.rank(axis=1)

i = 0
for r in r_names:
    ax[0,0].scatter(df_sum.loc[0.01,r], i, c='C0')
    ax[0,0].text(df_sum.loc[0.01,r], i, int(ranks.loc[0.01,r]), size=20)
    
    ax[0,1].scatter(df_sum.loc[0.025,r], i, c='C0')
    ax[0,1].text(df_sum.loc[0.025,r], i, int(ranks.loc[0.025,r]), size=20)
    
    ax[1,0].scatter(df_sum.loc[0.05,r], i, c='C0')
    ax[1,0].text(df_sum.loc[0.05,r], i, int(ranks.loc[0.05,r]), size=20)
    
    ax[1,1].scatter(df_sum.loc[0.1,r], i, c='C0')
    ax[1,1].text(df_sum.loc[0.1,r], i, int(ranks.loc[0.1,r]), size=20)
    i+= 1
        