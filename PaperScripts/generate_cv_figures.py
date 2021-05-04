# -*- coding: utf-8 -*-


from CST.utils.dataset_utils import load_sktime_dataset
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set()
sns.set_context("talk")

cv_path = r"C:\git_projects\CST\CV_10_results_10_[100, 95, 90, 85, 80].csv"
cv_f1 = r"C:\git_projects\CST\TESTF1_MEANS.csv"
cv_train =  r"C:\git_projects\CST\TRAINTrainTimes_MEDIANS.csv"
cv_test =  r"C:\git_projects\CST\TESTAvgPredTimes_MEDIANS.csv"
#cv_path = r"C:\Users\Antoine\Documents\git_projects\CST\CST\CV_10_results_10_[100, 95, 90, 85, 80].csv"
df = pd.read_csv(cv_path, sep=',').rename(columns={'Unnamed: 0': 'Dataset'})
df2 = pd.read_csv(cv_f1, sep=',').rename(columns={'TESTF1': 'Dataset'})
df3 = pd.read_csv(cv_train, sep=',').rename(columns={'TRAINTrainTimes': 'Dataset'})
df4 = pd.read_csv(cv_test, sep=',').rename(columns={'TESTAvgPredTimes': 'Dataset'})

df2 = df2[df2['Dataset'].isin(df['Dataset'])]
df3 = df3[df3['Dataset'].isin(df['Dataset'])]
df4 = df4[df4['Dataset'].isin(df['Dataset'])]

# In[]:
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], color='r')
ax.plot([0.0, 0.95], [0.05, 1.0], color='orange', linestyle='--', alpha=0.75)
ax.plot([0.05, 1.0], [0.0, 0.95], color='orange', linestyle='--', alpha=0.75)
x = df['MiniRKT_mean'].values
y = df['MiniCST_mean'].values
ax.scatter(x, y, s=25, alpha=0.75)
ax.set_ylabel('CST')
ax.set_xlabel('MiniRKT')
textstr = 'W - D - L (+/- 5%)\n {} - {} - {}'.format(sum(x*1.05 < y), sum((x*0.95<= y) & (y <= x*1.05)), sum(x > y*1.05))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

# In[]:
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], color='r')
ax.plot([0.0, 0.95], [0.05, 1.0], color='orange', linestyle='--', alpha=0.75)
ax.plot([0.05, 1.0], [0.0, 0.95], color='orange', linestyle='--', alpha=0.75)
x = df2['STC'].values
y = df[df['Dataset'].isin(df2['Dataset'])]['MiniCST_mean'].values
ax.scatter(x, y, s=25, alpha=0.75)
ax.set_ylabel('CST')
ax.set_xlabel('STC')
textstr = 'W - D - L (+/- 5%)\n {} - {} - {}'.format(sum(x*1.05 < y), sum((x*0.95<= y) & (y <= x*1.05)), sum(x > y*1.05))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

# In[]:

sizes = {}
for i, row in df.iterrows():
    if row['Dataset'] in df2['Dataset'].values:
        X, y, _ = load_sktime_dataset(row['Dataset'])
        sizes.update({row['Dataset'] : X.shape})

# In[]:
from sklearn.preprocessing import KBinsDiscretizer    

n_samples = np.array([sizes[k][0] for k in sizes])
k = KBinsDiscretizer(n_bins=15, strategy='quantile').fit(n_samples.reshape(-1,1))
x1 = k.inverse_transform(k.transform(n_samples.reshape(-1,1)))
y1 = np.array([df[df['Dataset']==k]['Runtime_MiniCST'] for k in sizes])
y1 = [np.mean(y1[np.where(x1==v)]) for v in np.unique(x1)]
y2 = np.array([df[df['Dataset']==k]['Runtime_MiniRKT'] for k in sizes])
y2 = [np.mean(y2[np.where(x1==v)]) for v in np.unique(x1)]
y3 = np.array([[df3[df3['Dataset']==k]['STC'] + df4[df4['Dataset']==k]['STC'] for k in sizes] for k in sizes])
y3 = [np.mean(y3[np.where(x1==v)]) for v in np.unique(x1)]
x = np.unique(x1)

fig, ax = plt.subplots(ncols=2,sharey=True,figsize=(15,6))
ax[0].set_yscale('log')
ax[0].plot(x, y1)
ax[0].plot(x, y2)
ax[0].plot(x, y3)
ax[0].set_title('Runtime for number of samples')
#TODO timedelta labels


n_timepoints = np.array([sizes[k][2] for k in sizes])
k = KBinsDiscretizer(n_bins=15, strategy='quantile').fit(n_timepoints.reshape(-1,1))
x1 = k.inverse_transform(k.transform(n_timepoints.reshape(-1,1)))
y1 = np.array([df[df['Dataset']==k]['Runtime_MiniCST'] for k in sizes])
y1 = [np.mean(y1[np.where(x1==v)]) for v in np.unique(x1)]
y2 = np.array([df[df['Dataset']==k]['Runtime_MiniRKT'] for k in sizes])
y2 = [np.mean(y2[np.where(x1==v)]) for v in np.unique(x1)]
y3 = np.array([[df3[df3['Dataset']==k]['STC'] + df4[df4['Dataset']==k]['STC'] for k in sizes] for k in sizes])
y3 = [np.mean(y3[np.where(x1==v)]) for v in np.unique(x1)]
x = np.unique(x1)

ax[1].set_yscale('log')
ax[1].plot(x, y1)
ax[1].plot(x, y2)
ax[1].plot(x, y3)
ax[1].set_title('Runtime for time series length')

# In[]:
df_latex = df[['Dataset','MiniCST_mean','MiniCST_std','MiniRKT_mean','MiniRKT_std']].astype(str)
df_latex['CST'] = df_latex['MiniCST_mean'].str[0:5] + ' (+/- '+  df_latex['MiniCST_std'].str[0:5]+')'
df_latex['Mini-ROCKET'] = df_latex['MiniRKT_mean'].str[0:5] + ' (+/- '+  df_latex['MiniRKT_std'].str[0:5]+')'
df_latex[['Dataset','CST','Mini-ROCKET']].to_latex('CV_table.tex',index=False)
