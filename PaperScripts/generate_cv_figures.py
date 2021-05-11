# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
sns.set_context("talk",font_scale=0.9)

#base_path = r"C:\Users\Antoine\Documents\git_projects\CST\CST\\"
base_path = r"C:\git_projects\CST\\"

cv_path = base_path + r"CV_10_results_10_[100, 95, 90, 85, 80]_final.csv"
cv_f1 = base_path + r"ucr_accuracy_LRS_versus_baselines.csv"

df = pd.read_csv(cv_path, sep=',').rename(columns={'Unnamed: 0': 'Dataset'})
df2 = pd.read_csv(cv_f1, sep=',').rename(columns={'dataset': 'Dataset'})

df = df[df['SFC_mean'] > 0]
df2 = df2[df2['Dataset'].isin(df['Dataset'])]

df_means = pd.concat([df[['Dataset','MiniCST_mean','MiniRKT_mean','SFC_mean']],df2[df2.columns.difference(['Dataset'])]],axis=1).rename(columns={'MiniCST_mean':'CST',
                                                                                                            'MiniRKT_mean':'MiniRKT',
                                                                                                            'SFC_mean':'SFC'})

# In[]:
competitors = ['MiniRKT', 'SFC', 'FS', 'LRS', 'LS', 'ST']   

ncols=3
nrows=2
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,11),sharey=True)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
for i, comp in enumerate(competitors):    
    ax[i//ncols,i%ncols].plot([0, 1], [0, 1], color='r')
    ax[i//ncols,i%ncols].plot([0.0, 0.95], [0.05, 1.0], color='orange', linestyle='--', alpha=0.75)
    ax[i//ncols,i%ncols].plot([0.05, 1.0], [0.0, 0.95], color='orange', linestyle='--', alpha=0.75)
    x = df_means[comp].values
    y = df_means['CST'].values
    ax[i//ncols,i%ncols].scatter(x, y, s=25, alpha=0.75)
    if i%ncols == 0:
        ax[i//ncols,i%ncols].set_ylabel('CST')
    ax[i//ncols,i%ncols].set_xlabel(comp)
    textstr = 'W - D - L (+/- 5%)\n {} - {} - {}'.format(sum(x*1.05 < y), sum((x*0.95<= y) & (y <= x*1.05)), sum(x > y*1.05))
    ax[i//ncols,i%ncols].text(0.05, 0.95, textstr, transform=ax[i//ncols,i%ncols].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

# In[]:
df_latex = df[['Dataset','MiniCST_mean','MiniCST_std','MiniRKT_mean','MiniRKT_std','SFC_mean','SFC_std']].astype(str)
df_latex['CST'] = df_latex['MiniCST_mean'].str[0:5] + ' (+/- '+  df_latex['MiniCST_std'].str[0:5]+')'
df_latex['Mini-ROCKET'] = df_latex['MiniRKT_mean'].str[0:5] + ' (+/- '+  df_latex['MiniRKT_std'].str[0:5]+')'
df_latex['SFC'] = df_latex['SFC_mean'].str[0:5] + ' (+/- '+  df_latex['SFC_std'].str[0:5]+')'
#TODO include STC
df_latex[['Dataset','CST','Mini-ROCKET', 'SFC']].to_latex(base_path+'CV_table.tex',index=False)
