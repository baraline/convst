# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import timedelta
sns.set()
sns.set_context('talk',font_scale=1.0)

base_path = "C:\git_projects\CST\\"
n_samples = base_path + "n_samples_Benchmark.csv"
n_timepoints = base_path + "tslength_Benchmark.csv"

#df = pd.read_csv(n_samples, sep=',')
df2 = pd.read_csv(n_timepoints, sep=',').set_index('Unnamed: 0')

df = pd.DataFrame(index=[10,100,1000,10000,100000],data={'CST':[0.5,1,2,3,4],'RKT':[0.1,0.5,1,1,2],'SFC':[10,20,30,40,100]})
#df2 = pd.DataFrame(index=[10,100,1000,10000,100000],data={'CST':[0.5,1,2,3,4],'RKT':[0.1,0.5,1,1,2],'SFC':[10,20,30,40,100]})
# In[]:
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
ax[1].set_yticks(ticks=[0.6,6,60,600])
ax[1].set_yticklabels(labels=['600ms','6s','1min','10min'])
ax[1].set_xlabel('number of timepoints')
ax[0].set_title('DucksAndGeese')

