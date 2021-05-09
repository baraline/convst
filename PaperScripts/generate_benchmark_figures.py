# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import timedelta
sns.set()
sns.set_context('talk',font_scale=1.0)

#base_path = "C:\Users\Antoine\Documents\git_projects\CST\CST\\"
base_path = "C:\git_projects\CST\\"
n_samples = base_path + "n_samples_Benchmark.csv"
n_timepoints = base_path + "tslength_Benchmark.csv"

#df = pd.read_csv(n_samples, sep=',')
df2 = pd.read_csv(n_timepoints, sep=',').set_index('Unnamed: 0')

df = pd.DataFrame(index=[10,100,1000,10000,100000],data={'CST':[0.5,1,2,3,4],'RKT':[0.1,0.5,1,1,2],'SFC':[10,20,30,40,100]})
#df2 = pd.DataFrame(index=[10,100,1000,10000,100000],data={'CST':[0.5,1,2,3,4],'RKT':[0.1,0.5,1,1,2],'SFC':[10,20,30,40,100]})
# In[]:
import matplotlib.patches as patches
fig, ax = plt.subplots(ncols=2, figsize=(15,5))
df.plot(ax=ax[0])
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_yticks(ticks=[0.6,6,60,600])
ax[0].set_yticklabels(labels=['600ms','6s','1min','10min'])
ax[0].set_xlabel('number of samples')
ax[0].set_title('InsectSound')


df2.plot(ax=ax[1])
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_yticks(ticks=[0.6,6,60,600,6000,60000,600000])
ax[1].set_yticklabels(labels=['600ms','6s','1min','10min','1h40','16h40','7j'])
ax[1].set_xlabel('number of timepoints')
ax[1].set_title('DucksAndGeese')

style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")
a3 = patches.FancyArrowPatch((df2.index[-1], df2.loc[df2.index[-1],'cst']), (df2.index[-1], df2.loc[df2.index[-1],'sfc']),
                             connectionstyle="arc3,rad=.15", **kw)

ax[1].add_patch(a3)
ax[1].text(df2.index[-1]-65000, (df2.loc[df2.index[-1],'cst']+df2.loc[df2.index[-1],'sfc'])//100,'x{}'.format(int(df2.loc[df2.index[-1],'sfc']//df2.loc[df2.index[-1],'cst'])))