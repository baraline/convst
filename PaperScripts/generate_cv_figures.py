# -*- coding: utf-8 -*-


from CST.base_transformers.convolutional_kernels import Rocket_kernel
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
from CST.utils.dataset_utils import load_sktime_dataset_split
from CST.utils.shapelets_utils import compute_distances, generate_strides_2D, generate_strides_1D
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
sns.set_context("talk")


cv_path = r"C:\Users\Antoine\Documents\git_projects\CST\CST\CV_10_results_10_[100, 95, 90, 85, 80].csv"
df = pd.read_csv(cv_path, sep=',').rename(columns={'Unnamed: 0':'Dataset'})

# In[]:
fig, ax = plt.subplots()
ax.plot([0, 1], [0,1], color='r')
x = df['MiniRKT_mean'].values
y = df['MiniCST_mean'].values
ax.scatter(x, y,s=25,alpha=0.75)
ax.set_ylabel('CST')
ax.set_xlabel('MiniRKT')
textstr = 'W - D - L\n{} - {} - {}'.format(sum(x<y), sum(x==y), sum(x>y))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)