# -*- coding: utf-8 -*-

from convst.transformers import R_DST
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")

# In[Class differences graph]:
fig, ax =plt.subplots(ncols=4, figsize=(20,5), sharey=True)
al = 0.75
scale = np.zeros((2,100))
scale[0, 20:40] = np.arange(0,1,0.05)
scale[1, 20:40] = np.arange(0,1,0.05)*2
ax[0].plot(scale[0],alpha=al, label='class 0')
ax[0].plot(scale[1],alpha=al, label='class 1')
ax[0].legend()
ax[0].set_title('Class difference by scale')

presence = np.zeros((2,100))
presence[0, 20:40] = np.arange(0,1,0.05)
presence[1, 20:40] = np.arange(0,1,0.05)[::-1]
ax[1].plot(presence[0],alpha=al, label='class 0')
ax[1].plot(presence[1],alpha=al, label='class 1')
ax[1].legend()
ax[1].set_title('Class Difference by motif')


location = np.zeros((2,100))
location[0, 20:40] = np.arange(0,1,0.05)
location[1, 60:80] = np.arange(0,1,0.05)
ax[2].plot(location[0],alpha=al, label='class 0')
ax[2].plot(location[1],alpha=al, label='class 1')
ax[2].legend()
ax[2].set_title('Class Difference by location')


location = np.zeros((2,100))
location[0, 10:30] = np.arange(0,1,0.05)
location[0, 65:85] = np.arange(0,1,0.05)
location[1, 10:30] = np.arange(0,1,0.05)
ax[3].plot(location[0],alpha=al, label='class 0')
ax[3].plot(location[1],alpha=al, label='class 1')
ax[3].legend()
ax[3].set_title('Class Difference by # occurence')

# In[dilated and non dilated shapelet]:
    
fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(15,3.5))
series_0 = np.zeros(100) 
series_1 = np.zeros(100) 
series_0[0:10] = np.arange(0,0.5,0.05)
series_0[10:30] = 0.5
series_0[30:40] = np.arange(0.5,1.0,0.05)
series_0[40:60] = 1.0
series_0[60:70] = np.arange(0.5,1.0,0.05)[::-1]
series_0[70:90] = 0.5
series_0[90:100] = np.arange(0,0.5,0.05)[::-1]

ax[0].plot(series_0, c='C0', alpha=0.75, linewidth=3)
ax[1].plot(series_0, c='C0', alpha=0.75, linewidth=3)
ax[0].scatter(np.arange(30)+5,series_0[5:35], c='C1', alpha=0.95)
ix = [5,25,45,65,85]
ax[1].scatter(ix, series_0[ix], c='C1', alpha=0.95)
ax[0].set_title('Shapelet of length 30')
ax[1].set_title('Shapelet of length 5 with a dilation of 20')
ax[1].axvline(25,0.48,0.25, linestyle='-.', c='C2')
ax[1].axvline(45,0.92,0.25, linestyle='-.', c='C2')
ax[1].arrow(25.5,0.22,19,0,width=0.005,color='C2',
            length_includes_head=True,head_width=0.05,
            head_length=3.0)
ax[1].text(28,0.24,'d=20',fontsize=18)
    
# In[]:    

from convst.utils.dataset_utils import load_sktime_dataset_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# In[]
X_train, X_test, y_train, y_test, _ = load_sktime_dataset_split(
    'Coffee', normalize=True)
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
y_train = y_train
print(X_train.shape)
print(X_test.shape)
# In[]: RD-ST
ac = R_DST(n_shapelets=10000, shapelet_sizes=[11]).fit(X_train, y_train)
x = ac.transform(X_train)
xt = ac.transform(X_test)
rf = make_pipeline(StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-6,6,20)))
rf.fit(x, y_train)
p = rf.predict(xt)
print(accuracy_score(y_test, p))

i_class = 0
ix = rf['ridgeclassifiercv'].coef_[i_class].argsort()
ii = 0
ac.visualise_one_shapelet(ix[ii]//3, X_test, y_test, i_class, figs=(17,12))
# In[]
#ac.visualise_one_shapelet(ix[ii]//3, X_test, y_test, i_class)
def class_vis(rdg, i_class, c):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(23,5))
    ix = np.zeros(rdg.coef_[0].shape[0])
    ix[1::3] = 1
    ix[2::3] = 2
    
    sns.boxplot(x=ix, y=rdg.coef_[i_class],ax=ax[0])
    ax[0].set_xticks([0,1,2])
    ax[0].set_xticklabels(['min','argmin','#match'])
    ax[0].set_ylabel('Ridge coef for class '+str(i_class))
    ax[1].set_ylabel('Ridge coef for class '+str(i_class))
    ax[1].set_xlabel('dilation')
    coef_sums = (rdg.coef_[i_class][0::3] + rdg.coef_[i_class][1::3] + rdg.coef_[i_class][2::3])/3
    sns.boxplot(x=c.dilation_, y=coef_sums,ax=ax[1])
    
    #sns.boxplot(x=c.length_, y=coef_sums,ax=ax[2])
    
class_vis(rf['ridgeclassifiercv'], 0, ac)
