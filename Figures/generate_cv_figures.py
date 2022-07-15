# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set()
sns.set_context("talk", font_scale=0.9)

base_path = "/home/lifo/Documents/git_projects/convst/results/"

baseline_path = base_path + r"CV_30_results_Random_final_(5_10).csv"
df = pd.read_csv(baseline_path, sep=',', index_col=0).rename(
    columns={"dataset": "Dataset"})
df = df[df['Dataset'] != '0']
sota_path = base_path + r"SOTA-AverageOver30.csv"
df2 = pd.read_csv(sota_path, sep=',').rename(columns={'TESTACC': 'Dataset'})
df2 = df2[df2['Dataset'].isin(df['Dataset'])]

df_info = pd.read_csv(base_path+'TSC_dataset_info.csv')

df_perf = pd.DataFrame(index=np.arange(df2.shape[0]), columns=['Dataset'])
df_perf_time = pd.DataFrame(index=np.arange(df2.shape[0]), columns=['Dataset'])
d_cols = df_info.columns.difference(['Dataset'])
a = 0
for i, grp in df.groupby('Dataset'):
    df_perf.loc[a, 'Dataset'] = i
    df_perf_time.loc[a, 'Dataset'] = i
    df_perf_time.loc[a, d_cols] = df_info.loc[df_info['Dataset']
                                              == i, d_cols].values[0]
    df_perf.loc[a, d_cols] = df_info.loc[df_info['Dataset']
                                         == i, d_cols].values[0]
    for j, d in grp.groupby('model'):
        df_perf_time.loc[a, j] = d['time_mean'].values[0]
        df_perf.loc[a, j] = d['acc_mean'].values[0]

    for j, serie in df2[df2['Dataset'] == i].items():
        df_perf.loc[a, j] = serie.values[0]
    a += 1

df_perf['total_len'] = df_perf['Train size'] * df_perf['Length']

# In[]:
m0 = 'RS Ridge v2'
m1 = 'TS-CHIEF'
margin = 0.05
margin /= 2
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
for f in ['total_len', 'Type']:
    fig, ax = plt.subplots(ncols=2, figsize=(15, 5), sharey=True, sharex=True)
    ax[0].scatter(df_perf[f], df_perf[m0])
    ax[0].set_title(m0)
    ax[1].scatter(df_perf[f], df_perf[m1])
    ax[1].set_title(m1)
    if f == 'Type':
        ax[0].set_xticks(df_perf[f].unique())
        ax[0].set_xticklabels(df_perf[f].unique(), rotation=-70)
        ax[1].set_xticks(df_perf[f].unique())
        ax[1].set_xticklabels(df_perf[f].unique(), rotation=-70)

    fig.suptitle(f)
    plt.show()
    diff = df_perf[m0]-df_perf[m1]
    plt.figure(figsize=(10, 5))
    plt.scatter(df_perf[f], diff)
    for j in range(df_perf.shape[0]):
        if abs(diff.loc[j]) > 0.05:
            plt.annotate(
                df_perf.loc[j, 'Dataset'],
                (df_perf.loc[j, f], diff[j]),
                fontsize=14,
                bbox=dict(boxstyle="round", alpha=0.1),
            )
    plt.title('diff {}'.format(f))
    textstr = 'W - D - L\n{} - {} - {}'.format(sum(diff > margin), sum(
        (-margin <= diff) & (diff <= margin)), sum(diff < -margin))
    plt.hlines(0, df_perf[f].min(), df_perf[f].max(), color='red')
    plt.text(0., 0.125, textstr, fontsize=14,
             verticalalignment='top', bbox=props)
    if f == 'Type':
        plt.xticks(df_perf[f].unique(), rotation=-70)
    plt.show()
# In[]:

baseline = 'RDST'
competitors = ['HC2', 'TS-CHIEF', 'HC1', 'InceptionTime']
#competitors = ['HIVE-COTE v1.0', 'RS Ridge', 'STC', 'InceptionTime']
ncols = 2
nrows = 2
limit_min = 0.2
limit_max = 1.0
margin = 0.05
margin /= 2
show_names = True
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 10), sharey=True)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
for i, comp in enumerate(competitors):
    if nrows >= 2:
        limp = (limit_max-limit_min)*0.05
        ax[i // ncols, i % ncols].set_xlim(limit_min-limp, limit_max+limp)
        ax[i // ncols, i % ncols].set_ylim(limit_min-limp, limit_max+limp)
        ax[i // ncols, i %
            ncols].plot([limit_min, limit_max], [limit_min, limit_max], color='r')
        ax[i // ncols, i % ncols].plot([limit_min, limit_max-margin], [
                                       limit_min + margin, limit_max], color='orange', linestyle='--', alpha=0.75)
        ax[i // ncols, i % ncols].plot([limit_min + margin, limit_max], [
                                       limit_min, limit_max-margin], color='orange', linestyle='--', alpha=0.75)
        ax[i // ncols, i % ncols].annotate(
            '{}'.format(margin),
            (limit_min + margin/2 + 0.0125, limit_min+margin/2 - 0.01),
            fontsize=12
        )
        x = df_perf[comp].values
        y = df_perf[baseline].values
        ax[i // ncols, i % ncols].scatter(x, y, s=15, alpha=0.75, zorder=2)
        if show_names:
            for j in range(df_perf.shape[0]):
                x1 = df_perf.loc[j, comp]
                y1 = df_perf.loc[j, baseline]
                if abs(x1 - y1) > margin*5:
                    ax[i // ncols, i % ncols].annotate(
                        "{} ({})".format(
                            df_perf.loc[j, 'Dataset'], df_perf.loc[j, 'Type']),
                        (x1, y1),
                        fontsize=14,
                        bbox=dict(boxstyle="round", alpha=0.1),
                    )
        if i % ncols == 0:
            ax[i // ncols, i % ncols].set_ylabel(baseline)
        ax[i // ncols, i % ncols].set_xlabel(comp)

        textstr = 'W - D - L\n{} - {} - {}'.format(sum(x+margin < y), sum(
            (x <= y+margin) & (y-margin <= x)), sum(x-margin > y))
        ax[i // ncols, i % ncols].text(0.05, 0.95, textstr, transform=ax[i // ncols, i % ncols].transAxes, fontsize=14,
                                       verticalalignment='top', bbox=props)
# In[]:

model_cols = df_perf.columns.difference(
    ['min', 'max', 'Dataset', 'total_len', 'Length', 'Test size', 'Train size', 'Type', 'n_classes'])
df_perf['min'] = df_perf[model_cols].min(axis=1)
df_perf['max'] = df_perf[model_cols].max(axis=1)
df_perf = df_perf.sort_values('Type').reset_index(drop=True)
df_perf.loc[df_perf['Type'] == 'HEMODYNAMICS', 'Type'] = 'HEMO'

# In[]

target = 'RDST'
comp = 'HC2'
plt.figure(figsize=(40, 10))
plt.plot(df_perf['Dataset'], df_perf[target], c='C1', label=target)
plt.plot(df_perf['Dataset'], df_perf[comp], c='C4', label=comp)
plt.fill_between(df_perf['Dataset'], df_perf['min'], df_perf['max'], alpha=0.4)
plt.xticks(rotation=-90)
plt.ylim(df_perf['min'].min(), 1.2)
plt.tick_params(axis='x', which='major', labelsize=20)
b = 0.05
for ib, t in enumerate(df_perf['Type'].unique()):
    d = df_perf[df_perf['Type'] == t]
    plt.bar(d.index.values[0]-0.5, 1.2, width=0.25,
            linestyle='-.', color='C2', alpha=0.75)
    plt.bar(d.index.values[-1]+0.5, 1.2, width=0.25,
            linestyle='-.', color='C2', alpha=0.75)
    xi = (d.index.values[-1] - d.index.values[0])/2
    plt.text(d.index.values[0]+xi, 1.1+(b*(ib % 2)),
             t,  ha="center", va="center", size=20)
mask = df_perf['max'] == df_perf[target]
#plt.scatter(df_perf.loc[mask,'Dataset'], df_perf.loc[mask,target], color='red')
plt.legend(loc='lower center')
plt.tight_layout()
