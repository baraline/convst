# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:06:19 2021

@author: A694772
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Switched from Latex to Markdown for github display

base_path = r"C:/git_projects/CST/benchmarks/"
df_path = base_path+"df_perf.csv"
df = pd.read_csv(df_path,index_col=0)
tuples = np.asarray([df['Version'].unique(),df['Dataset'].unique()])

fig, ax = plt.subplots(ncols=3,figsize=(19,5))

#####################################
#                                   #
#             F1 Score              #
#                                   #
#####################################

df_latex_f1 = pd.DataFrame(columns=["F1 Score Ridge","F1 Score RF"], index=pd.MultiIndex.from_product(tuples, names=["version", "dataset"]))
for version, df_vers in df.groupby('Version'):
    for name, df_name in df_vers.groupby('Dataset'):
        df_latex_f1.loc[(version, name),"F1 Score Ridge"] = df_name[df_name["Algorithm"] == "CST + Ridge"]["F1_score"].values[0]
        df_latex_f1.loc[(version, name),"F1 Score RF"] = df_name[df_name["Algorithm"] == "CST + RF"]["F1_score"].values[0]

df_latex_f1.to_markdown(base_path+'perf_f1.md', index=True)

df_latex_f1["F1 Score Ridge"] = df_latex_f1["F1 Score Ridge"].apply(lambda x : float(x.split('(')[0]))
df_latex_f1["F1 Score RF"] = df_latex_f1["F1 Score RF"].apply(lambda x : float(x.split('(')[0]))


df_latex_f1.groupby(level='version').mean().plot(title="Mean F1 Score per version", ax=ax[0],
                                                 xlabel='Version',legend=False)


#####################################
#                                   #
#             Acc Score             #
#                                   #
#####################################


df_latex_acc = pd.DataFrame(columns=["Acc Score Ridge","Acc Score RF"], index=pd.MultiIndex.from_product(tuples, names=["version", "dataset"]))
for version, df_vers in df.groupby('Version'):
    for name, df_name in df_vers.groupby('Dataset'):
        df_latex_acc.loc[(version, name),"Acc Score Ridge"] = df_name[df_name["Algorithm"] == "CST + Ridge"]["Accuracy_score"].values[0]
        df_latex_acc.loc[(version, name),"Acc Score RF"] = df_name[df_name["Algorithm"] == "CST + RF"]["Accuracy_score"].values[0]

df_latex_acc.to_markdown(base_path+'perf_acc.md', index=True)

df_latex_acc["Acc Score Ridge"] = df_latex_acc["Acc Score Ridge"].apply(lambda x : float(x.split('(')[0]))
df_latex_acc["Acc Score RF"] = df_latex_acc["Acc Score RF"].apply(lambda x : float(x.split('(')[0]))


df_latex_acc.groupby(level='version').mean().plot(title="Mean Acc Score per version",ax=ax[1],
                                                       xlabel='Version',legend=False)

#####################################
#                                   #
#             Run time              #
#                                   #
#####################################


df_latex_time = pd.DataFrame(columns=["RunTime Ridge","RunTime RF"], index=pd.MultiIndex.from_product(tuples, names=["version", "dataset"]))
for version, df_vers in df.groupby('Version'):
    for name, df_name in df_vers.groupby('Dataset'):
        df_latex_time.loc[(version, name),"RunTime Ridge"] = df_name[df_name["Algorithm"] == "CST + Ridge"]["RunTime"].values[0]
        df_latex_time.loc[(version, name),"RunTime RF"] = df_name[df_name["Algorithm"] == "CST + RF"]["RunTime"].values[0]

df_latex_time.to_markdown(base_path+'perf_runtime.md', index=True)

df_latex_time["RunTime Ridge"] = df_latex_time["RunTime Ridge"].apply(lambda x : float(x.split('(')[0]))
df_latex_time["RunTime RF"] = df_latex_time["RunTime RF"].apply(lambda x : float(x.split('(')[0]))


df_latex_time.groupby(level='version').mean().plot(title="Mean RunTime per version",ax=ax[2],
                                                        xlabel='Version')

plt.show()
plt.savefig(base_path+"perf_fig.png")

