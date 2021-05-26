# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
sns.set_context("talk", font_scale=0.9)

#base_path = r"C:\Users\Antoine\Documents\git_projects\CST\CST\csv_results\\"
base_path = r"C:\git_projects\CST\csv_results\\"

cv_path = base_path + r"CV_30_results_(200,1.0)_9_80.csv"
cv_f1 = base_path + r"TESTF1_MEANS.csv"
cv_f1_std = base_path + r"TESTF1_STDDEV.csv"

df = pd.read_csv(cv_path, sep=',').rename(columns={'Unnamed: 0': 'Dataset'})
df4 = pd.read_csv(base_path + 'CV_30_results_(200,1.0)_11_80.csv', sep=',').rename(columns={'Unnamed: 0': 'Dataset'})
df2 = pd.read_csv(cv_f1, sep=',').rename(columns={'TESTF1': 'Dataset'})
df3 = pd.read_csv(cv_f1_std, sep=',').rename(
    columns={'TESTF1STDDEVS': 'Dataset'})

colcst = ['CST_f1_mean','CST_f1_std','CST_acc_mean','CST_acc_std']
df[colcst] = df4[colcst]

df = df[(df['CST_f1_mean'] > 0) & (df['MiniRKT_f1_mean'] > 0)]
df2 = df2[df2['Dataset'].isin(df['Dataset'])]
df3 = df3[df3['Dataset'].isin(df['Dataset'])]

df_means = pd.concat([df[['Dataset', 'MiniRKT_f1_mean', 'CST_f1_mean', 'SFC_f1_mean', 'MrSEQL_f1_mean']], df2[df2.columns.difference(['Dataset'])]], axis=1).rename(columns={'CST_f1_mean': 'CST',
                                                                                                                                                  'MiniRKT_f1_mean': 'MiniRKT',
                                                                                                                                                  'SFC_f1_mean': 'SFC',
                                                                                                                                                  'MrSEQL_f1_mean':'MrSEQL'})
competitors = ['MiniRKT', 'SFC', 'STC', 'MrSEQL']
ncols = 2
nrows = 2
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10), sharey=True)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
for i, comp in enumerate(competitors):
    ax[i // ncols,i % ncols].plot([0, 1], [0, 1], color='r')
    #ax[i%ncols].plot([0.0, 0.95], [0.05, 1.0], color='orange', linestyle='--', alpha=0.75)
    #ax[i%ncols].plot([0.05, 1.0], [0.0, 0.95], color='orange', linestyle='--', alpha=0.75)
    x = df_means[comp].values
    y = df_means['CST'].values
    ax[i // ncols,i % ncols].scatter(x, y, s=25, alpha=0.75)
    if i % ncols == 0:
        ax[i // ncols,i % ncols].set_ylabel('CST')
    ax[i // ncols,i % ncols].set_xlabel(comp)
    #textstr = 'W - D - L (+/- 5%)\n{} - {} - {}'.format(sum(x*1.05 < y), sum((x*0.95<= y) & (y <= x*1.05)), sum(x > y*1.05))
    textstr = 'W - D - L\n{} - {} - {}'.format(
        sum(x < y), sum((x <= y) & (y <= x)), sum(x > y))
    ax[i // ncols,i % ncols].text(0.05, 0.95, textstr, transform=ax[i // ncols,i % ncols].transAxes, fontsize=14,
                       verticalalignment='top', bbox=props)
    
ranking_path = r"params_csv_all.csv"
df_params = pd.read_csv(base_path+ranking_path).rename(columns={'Unnamed: 0': 'Dataset'})

#df2['STC'] = df2['STC'].apply(lambda x: '%.5f' % x)
#df3['STC'] = df3['STC'].apply(lambda x: '%.5f' % x)
df_latex = df[['Dataset', 'CST_f1_mean', 'CST_f1_std', 'MiniRKT_f1_mean',
               'MiniRKT_f1_std', 'SFC_f1_mean', 'SFC_f1_std','MrSEQL_f1_mean','MrSEQL_f1_std']]
df_latex['STC_f1_mean'] = df2['STC']
df_latex['STC_f1_std'] = df3['STC']

df_latex2 = df_latex.set_index('Dataset')
df_latex['Dataset'] = df_latex['Dataset'].apply(lambda x: x+'*' if x in df_params['Dataset'].values else x)
df_latex['STC_f1_mean'] = df_latex['STC_f1_mean'].apply(lambda x: '%.5f' % x)
df_latex['STC_f1_std'] = df_latex['STC_f1_std'].apply(lambda x: '%.5f' % x)
df_latex = df_latex.astype(str)

df_latex['CST'] = df_latex['CST_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex['CST_f1_std'].str[0:5]+')'
df_latex['Mini-ROCKET'] = df_latex['MiniRKT_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex['MiniRKT_f1_std'].str[0:5]+')'
df_latex['SFC'] = df_latex['SFC_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex['SFC_f1_std'].str[0:5]+')'
df_latex['MrSEQL'] = df_latex['MrSEQL_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex['MrSEQL_f1_std'].str[0:5]+')'
df_latex['STC'] = df_latex['STC_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex['STC_f1_std'].str[0:5]+')'

df_latex[['Dataset', 'CST', 'Mini-ROCKET', 'SFC', 'MrSEQL', 'STC']
         ].to_latex(base_path+'CV_table_f1.tex', index=False)


df_type = pd.read_csv(base_path+'dataset_type.csv').set_index('Dataset')

df_latex2['Type'] = df_type[' Type']
counts = df_latex2.groupby('Type').count()['CST_f1_mean']
df_latex3 = df_latex2.groupby('Type').std()
df_latex2 = df_latex2.groupby('Type').mean()

df_latex2 = df_latex2.reset_index()
df_latex3 = df_latex3.reset_index()

df_latex2['Type'] = df_latex2['Type'].apply(lambda x : x + ' (' +str(counts.loc[x])+ ')')
df_latex2['STC_f1_mean'] = df_latex2['STC_f1_mean'].apply(lambda x: '%.5f' % x)
df_latex2['STC_f1_std'] = df_latex2['STC_f1_std'].apply(lambda x: '%.5f' % x)
df_latex2 = df_latex2.astype(str)
df_latex3 = df_latex3.astype(str)

df_latex2['CST'] = df_latex2['CST_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['CST_f1_mean'].str[0:5]+')'
df_latex2['SFC'] = df_latex2['SFC_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['SFC_f1_mean'].str[0:5]+')'
df_latex2['MrSEQL'] = df_latex2['MrSEQL_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['MrSEQL_f1_mean'].str[0:5]+')'
df_latex2['STC'] = df_latex2['STC_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['STC_f1_mean'].str[0:5]+')'
df_latex2['Mini-ROCKET'] = df_latex2['MiniRKT_f1_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['MiniRKT_f1_mean'].str[0:5]+')'
df_latex2 = df_latex2.drop(df_latex2.columns.difference(['Type','CST','Mini-ROCKET','STC','SFC','MrSEQL']),axis=1)
df_latex2.to_latex(base_path+'CV_table_type.tex', index=False)
# In[]:

cv_path = base_path + r"CV_30_results_(200,1.0)_9_80.csv"
cv_acc = base_path + r"TESTACC_MEANS.csv"
cv_acc_std = base_path + r"TESTACC_STDDEV.csv"

df = pd.read_csv(cv_path, sep=',').rename(columns={'Unnamed: 0': 'Dataset'})

df2 = pd.read_csv(cv_acc, sep=',').rename(columns={'TESTACC': 'Dataset'})
df3 = pd.read_csv(cv_acc_std, sep=',').rename(
    columns={'TESTACCSTDDEVS': 'Dataset'})
df4 = pd.read_csv(base_path + 'CV_30_results_(200,1.0)_11_80.csv', sep=',').rename(columns={'Unnamed: 0': 'Dataset'})

colcst = ['CST_f1_mean','CST_f1_std','CST_acc_mean','CST_acc_std']
df[colcst] = df4[colcst]

df = df[(df['CST_f1_mean'] > 0) & (df['MiniRKT_f1_mean'] > 0)]
df2 = df2[df2['Dataset'].isin(df['Dataset'])]
df3 = df3[df3['Dataset'].isin(df['Dataset'])]

df_means = pd.concat([df[['Dataset', 'MiniRKT_acc_mean', 'CST_acc_mean', 'SFC_acc_mean', 'MrSEQL_acc_mean']], 
                      df2[df2.columns.difference(['Dataset'])]], axis=1).rename(columns={'CST_acc_mean': 'CST',
                                                                                        'MiniRKT_acc_mean': 'MiniRKT',
                                                                                        'SFC_acc_mean': 'SFC',
                                                                                        'MrSEQL_acc_mean':'MrSEQL'}).reset_index(drop=True)
competitors = ['MiniRKT', 'SFC', 'STC', 'MrSEQL']
ncols = 2
nrows = 2
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10), sharey=True)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
for i, comp in enumerate(competitors):
    ax[i // ncols,i % ncols].plot([0, 1], [0, 1], color='r')
    #ax[i%ncols].plot([0.0, 0.95], [0.05, 1.0], color='orange', linestyle='--', alpha=0.75)
    #ax[i%ncols].plot([0.05, 1.0], [0.0, 0.95], color='orange', linestyle='--', alpha=0.75)
    x = df_means[comp].values
    y = df_means['CST'].values
    ax[i // ncols,i % ncols].scatter(x, y, s=25, alpha=0.75)
    if i % ncols == 0:
        ax[i // ncols,i % ncols].set_ylabel('CST')
    ax[i // ncols,i % ncols].set_xlabel(comp)
    #textstr = 'W - D - L (+/- 5%)\n{} - {} - {}'.format(sum(x*1.05 < y), sum((x*0.95<= y) & (y <= x*1.05)), sum(x > y*1.05))
    textstr = 'W - D - L\n{} - {} - {}'.format(
        sum(x < y), sum((x <= y) & (y <= x)), sum(x > y))
    ax[i // ncols,i % ncols].text(0.05, 0.95, textstr, transform=ax[i // ncols,i % ncols].transAxes, fontsize=14,
                       verticalalignment='top', bbox=props)
    
df_latex = df[['Dataset', 'CST_acc_mean', 'CST_acc_std', 'MiniRKT_acc_mean',
               'MiniRKT_acc_std', 'SFC_acc_mean', 'SFC_acc_std','MrSEQL_acc_mean','MrSEQL_acc_std']]
df_latex['STC_acc_mean'] = df2['STC']
df_latex['STC_acc_std'] = df3['STC']

df_latex2 = df_latex.set_index('Dataset')
df_latex['Dataset'] = df_latex['Dataset'].apply(lambda x: x+'*' if x in df_params['Dataset'].values else x)
df_latex['STC_acc_mean'] = df_latex['STC_acc_mean'].apply(lambda x: '%.5f' % x)
df_latex['STC_acc_std'] = df_latex['STC_acc_std'].apply(lambda x: '%.5f' % x)
df_latex = df_latex.astype(str)

df_latex['CST'] = df_latex['CST_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex['CST_acc_std'].str[0:5]+')'
df_latex['Mini-ROCKET'] = df_latex['MiniRKT_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex['MiniRKT_acc_std'].str[0:5]+')'
df_latex['SFC'] = df_latex['SFC_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex['SFC_acc_std'].str[0:5]+')'
df_latex['MrSEQL'] = df_latex['MrSEQL_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex['MrSEQL_acc_std'].str[0:5]+')'
df_latex['STC'] = df_latex['STC_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex['STC_acc_std'].str[0:5]+')'

df_latex[['Dataset', 'CST', 'Mini-ROCKET', 'SFC', 'MrSEQL', 'STC']
         ].to_latex(base_path+'CV_table_acc.tex', index=False)


df_type = pd.read_csv(base_path+'dataset_type.csv').set_index('Dataset')

df_latex2['Type'] = df_type[' Type']
counts = df_latex2.groupby('Type').count()['CST_acc_mean']
df_latex3 = df_latex2.groupby('Type').std().fillna(0)
df_latex2 = df_latex2.groupby('Type').mean()

df_latex2 = df_latex2.reset_index()
df_latex3 = df_latex3.reset_index()

df_latex2['Type'] = df_latex2['Type'].apply(lambda x : x + ' (' +str(counts.loc[x])+ ')')
df_latex2['STC_acc_mean'] = df_latex2['STC_acc_mean'].apply(lambda x: '%.5f' % x)
df_latex2['STC_acc_std'] = df_latex2['STC_acc_std'].apply(lambda x: '%.5f' % x)
df_latex2 = df_latex2.astype(str)
df_latex3 = df_latex3.astype(str)

df_latex2['CST'] = df_latex2['CST_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['CST_acc_mean'].str[0:5]+')'
df_latex2['SFC'] = df_latex2['SFC_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['SFC_acc_mean'].str[0:5]+')'
df_latex2['MrSEQL'] = df_latex2['MrSEQL_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['MrSEQL_acc_mean'].str[0:5]+')'
df_latex2['STC'] = df_latex2['STC_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['STC_acc_mean'].str[0:5]+')'
df_latex2['Mini-ROCKET'] = df_latex2['MiniRKT_acc_mean'].str[0:5] + \
    ' (+/- ' + df_latex3['MiniRKT_acc_mean'].str[0:5]+')'
df_latex2 = df_latex2.drop(df_latex2.columns.difference(['Type','CST','Mini-ROCKET','STC','SFC','MrSEQL']),axis=1)
df_latex2.to_latex(base_path+'CV_table_type.tex', index=False)