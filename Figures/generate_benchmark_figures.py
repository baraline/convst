# -*- coding: utf-8 -*-

import matplotlib.patches as patches
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
models = ['RDST','Rocket','DrCIF','HC1','STC']
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