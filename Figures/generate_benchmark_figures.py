# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
sns.set_context('talk', font_scale=1.0)

base_path = r"C:\Users\a694772\OneDrive - Worldline\Documents\git_projects\convst\results\\"
n_samples = base_path + "n_samples_benchmarks.csv"
n_timepoints = base_path + "n_timepoints_benchmarks.csv"

df = pd.read_csv(n_samples, sep=',', index_col=0)
df2 = pd.read_csv(n_timepoints, sep=',', index_col=0)
models = ['RDST','Rocket','DrCIF','HC1','HC2','STC']
# In[]:
fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

df[models].plot(ax=ax[0], legend=False)

ax[0].set_yscale('log')
#ax[0].set_xscale('log')
ax[0].set_yticks(ticks=[0.6, 6, 60, 600, 6000, 60000])
ax[0].set_yticklabels(labels=['600ms', '6s', '1min', '10min', '1h40', '16h40'])
ax[0].set_xlabel('number of samples')
ax[0].set_title('Crop')

style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="black")
i_mod = 0


df2[models].plot(ax=ax[1], legend=False)
ax[1].set_yscale('log')
#ax[1].set_xscale('log')
ax[1].set_yticks(ticks=[0.6, 6, 60, 600, 6000, 60000])
ax[1].set_yticklabels(labels=['600ms', '6s', '1min',
                              '10min', '1h40', '16h40'])
ax[1].set_xlabel('number of timepoints')
ax[1].set_title('Rock')

# In[]:
    
cols = ['RDST','RDST sub=0.5 alpha=0.5', 'RDST sub=0.5 alpha=1.0','RDST + alpha=0.5', 'RDST + alpha=1.0']

df_perf['Total size'] = (df_perf['Train size'] + df_perf['Test size'])*df_perf['Length']
s = 'Total size'
bins=20
bin_sizes = (df_perf[s].max() - df_perf[s].min()) / bins
bin_ranges = np.arange(df_perf[s].min(), df_perf[s].max()+bin_sizes, bin_sizes)
barWidth = 0.17
 
# Choose the height of the blue bars

bars = np.asarray([
    [df_perf.loc[df_perf[s].between(bin_ranges[i],bin_ranges[i+1]),col].sum() 
     for i in range(bins)] for col in cols]
)
#idx = np.unique(np.where(~np.isnan(bars))[1])
idx = np.unique(np.where(bars!=0)[1])
r = np.asarray([x for x in range(len(idx))])

plt.figure(figsize=(16,8))
for i, col in enumerate(cols):
    plt.bar(r + i*barWidth, bars[i,idx], width = barWidth, label=col)
  
# general layout
plt.xticks(
    [r + (len(cols)*barWidth)/2 for r in range(len(idx))],
    ["[{}, {}]".format(int(bin_ranges[i]), int(bin_ranges[i+1])) for i in idx],
    rotation=-80
)
plt.ylabel('total fit + predict time in seconds')
plt.xlabel('Number of samples x number of timestamps')
plt.legend() 
# Show graphic
plt.show()
