# -*- coding: utf-8 -*-


from CST.utils.dataset_utils import load_sktime_dataset
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set()
sns.set_context("talk")

base_path = r"C:\Users\Antoine\Documents\git_projects\CST\CST\\"
cv_path = base_path + r"CV_10_results_10_[100, 95, 90, 85, 80]_noccp.csv"
cv_f1 = base_path + r"TESTF1_MEANS.csv"
cv_train = base_path +  r"TRAINTrainTimes_MEDIANS.csv"
cv_test =  base_path + r"TESTAvgPredTimes_MEDIANS.csv"

df = pd.read_csv(cv_path, sep=',').rename(columns={'Unnamed: 0': 'Dataset'})
df2 = pd.read_csv(cv_f1, sep=',').rename(columns={'TESTF1': 'Dataset'})
df2 = df2[df2['Dataset'].isin(df['Dataset'])]


# In[]:
fig, ax = plt.subplots(ncols=2, figsize=(15,5))
ax[0].plot([0, 1], [0, 1], color='r')
ax[0].plot([0.0, 0.95], [0.05, 1.0], color='orange', linestyle='--', alpha=0.75)
ax[0].plot([0.05, 1.0], [0.0, 0.95], color='orange', linestyle='--', alpha=0.75)
x = df['MiniRKT_mean'].values
y = df['MiniCST_mean'].values
ax[0].scatter(x, y, s=25, alpha=0.75)
ax[0].set_ylabel('CST')
ax[0].set_xlabel('MiniRKT')
textstr = 'W - D - L (+/- 5%)\n {} - {} - {}'.format(sum(x*1.05 < y), sum((x*0.95<= y) & (y <= x*1.05)), sum(x > y*1.05))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

# place a text box in upper left in axes coords
ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax[1].plot([0, 1], [0, 1], color='r')
ax[1].plot([0.0, 0.95], [0.05, 1.0], color='orange', linestyle='--', alpha=0.75)
ax[1].plot([0.05, 1.0], [0.0, 0.95], color='orange', linestyle='--', alpha=0.75)
x = df2['STC'].values
y = df[df['Dataset'].isin(df2['Dataset'])]['MiniCST_mean'].values
ax[1].scatter(x, y, s=25, alpha=0.75)
ax[1].set_ylabel('CST')
ax[1].set_xlabel('STC')
textstr = 'W - D - L (+/- 5%)\n {} - {} - {}'.format(sum(x*1.05 < y), sum((x*0.95<= y) & (y <= x*1.05)), sum(x > y*1.05))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

# place a text box in upper left in axes coords
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

# In[]:
df_latex = df[['Dataset','MiniCST_mean','MiniCST_std','MiniRKT_mean','MiniRKT_std']].astype(str)
df_latex['CST'] = df_latex['MiniCST_mean'].str[0:5] + ' (+/- '+  df_latex['MiniCST_std'].str[0:5]+')'
df_latex['Mini-ROCKET'] = df_latex['MiniRKT_mean'].str[0:5] + ' (+/- '+  df_latex['MiniRKT_std'].str[0:5]+')'
df_latex[['Dataset','CST','Mini-ROCKET']].to_latex('CV_table.tex',index=False)
