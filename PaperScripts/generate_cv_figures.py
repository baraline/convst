# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
sns.set_context("talk", font_scale=0.9)

base_path = r"C:\Users\Antoine\Documents\git_projects\CST\CST\csv_results\\"
#base_path = r"C:\git_projects\CST\csv_results\\"

cv_path = base_path + r"CV_30_results_(200,1.0)_9_80.csv"
cv_f1 = base_path + r"TESTF1_MEANS.csv"
cv_f1_std = base_path + r"TESTF1_STDDEV.csv"

df = pd.read_csv(cv_path, sep=',').rename(columns={'Unnamed: 0': 'Dataset'})

df2 = pd.read_csv(cv_f1, sep=',').rename(columns={'TESTF1': 'Dataset'})
df3 = pd.read_csv(cv_f1_std, sep=',').rename(
    columns={'TESTF1STDDEVS': 'Dataset'})

df = df[df['CST_f1_mean'] > 0]
df2 = df2[df2['Dataset'].isin(df['Dataset'])]
df3 = df3[df3['Dataset'].isin(df['Dataset'])]

# In[]:

df_means = pd.concat([df[['Dataset', 'MiniRKT_f1_mean', 'CST_f1_mean', 'SFC_f1_mean', 'MrSEQL_f1_mean']], df2[df2.columns.difference(['Dataset'])]], axis=1).rename(columns={'CST_f1_mean': 'CST',
                                                                                                                                                  'MiniRKT_f1_mean': 'MiniRKT',
                                                                                                                                                  'SFC_f1_mean': 'SFC',
                                                                                                                                                  'MrSEQL_f1_mean':'MrSEQL'})
competitors = ['MiniRKT', 'SFC', 'STC', 'MrSEQL']
ncols = 4
nrows = 1
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 5), sharey=True)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
for i, comp in enumerate(competitors):
    ax[i % ncols].plot([0, 1], [0, 1], color='r')
    #ax[i%ncols].plot([0.0, 0.95], [0.05, 1.0], color='orange', linestyle='--', alpha=0.75)
    #ax[i%ncols].plot([0.05, 1.0], [0.0, 0.95], color='orange', linestyle='--', alpha=0.75)
    x = df_means[comp].values
    y = df_means['CST'].values
    ax[i % ncols].scatter(x, y, s=25, alpha=0.75)
    if i % ncols == 0:
        ax[i % ncols].set_ylabel('CST')
    ax[i % ncols].set_xlabel(comp)
    #textstr = 'W - D - L (+/- 5%)\n{} - {} - {}'.format(sum(x*1.05 < y), sum((x*0.95<= y) & (y <= x*1.05)), sum(x > y*1.05))
    textstr = 'W - D - L\n{} - {} - {}'.format(
        sum(x < y), sum((x <= y) & (y <= x)), sum(x > y))
    ax[i % ncols].text(0.05, 0.95, textstr, transform=ax[i % ncols].transAxes, fontsize=14,
                       verticalalignment='top', bbox=props)

# In[]:
df_latex = df[['Dataset', 'CST_mean', 'CST_std', 'MiniRKT_mean',
               'MiniRKT_std', 'SFC_mean', 'SFC_std']].astype(str)
df_latex['CST'] = df_latex['CST_mean'].str[0:5] + \
    ' (+/- ' + df_latex['CST_std'].str[0:5]+')'
df_latex['Mini-ROCKET'] = df_latex['MiniRKT_mean'].str[0:5] + \
    ' (+/- ' + df_latex['MiniRKT_std'].str[0:5]+')'
df_latex['SFC'] = df_latex['SFC_mean'].str[0:5] + \
    ' (+/- ' + df_latex['SFC_std'].str[0:5]+')'
df_latex['STC'] = df2['STC'].astype(
    str).str[0:5] + ' (+/- ' + df3['STC'].astype(str).str[0:5]+')'
df_latex[['Dataset', 'CST', 'Mini-ROCKET', 'SFC', 'STC']
         ].to_latex(base_path+'CV_table.tex', index=False)
