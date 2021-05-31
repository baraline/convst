# -*- coding: utf-8 -*-

import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from CST.utils.dataset_utils import load_sktime_dataset_split

# Load Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'GunPoint', normalize=True)

# In[]:
# First run will be slow due to numba compilations on the first call. Run small dataset like GunPoint the first time !
# Put verbose = 1 to see some of the progression of the algorithm.

cst = make_pipeline(
    ConvolutionalShapeletTransformer(verbose=0),
    RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                      normalize=True, class_weight='balanced')
)

cst.fit(X_train, y_train)
pred = cst.predict(X_test)
print("Accuracy Score for CST : {}".format(accuracy_score(y_test, pred)))
print("Accuracy Score for CST : {}".format(
    f1_score(y_test, pred, average='macro')))

"""
sns.set(context='talk')
n_classes = np.unique(y_train).shape[0]
topk = 2
sample_id = 3
print(y_test[sample_id])

i_ft = np.argsort(rf.feature_importances_)[::-1][0:topk].astype(str)
fig, ax = plt.subplots(nrows=topk, ncols=3, figsize=(10*n_classes, 5*topk))
df = pd.DataFrame(X_cst_train)
df.columns = df.columns.astype(str)
df['y'] = y_train

for i in range(topk):
    df_long = pd.melt(df[[i_ft[i], "y"]], "y", var_name=" ", value_name="")
    sns.boxplot(x=" ", hue="y", y="", data=df_long, ax=ax[i, 0], linewidth=2.5)
    ax[i, 0].axhline(X_cst_test[sample_id, int(i_ft[i])],
                     color='red', linestyle='--')
    ax[0, 0].set_title("BoxPlot of 1/d for training samples")
    ax[0, 0].set_xlabel("Shapelet n° {}".format(i_ft[0]))
    ax[0, 1].set_title("Shapelet")
    ax[0, 2].set_title("Test sample with scaled shapelet")
    ax[i, 2].plot(X_test[sample_id][0])
    dil = cst.dil[int(i_ft[i])]
    shp = cst.shp[int(i_ft[i])]
    x = generate_strides_1D(X_test[sample_id][0], 9, dil)
    x = (x - x.mean(axis=-1, keepdims=True))/x.std(axis=-1, keepdims=True)
    d = cdist(x, shp.reshape(1, 9), metric='sqeuclidean')
    loc = d.argmin()
    x_indexes = [loc + j*dil for j in range(9)]
    shp_v = (shp * X_test[sample_id, 0, x_indexes].std()
             ) + X_test[sample_id, 0, x_indexes].mean()
    ax[i, 2].scatter(x_indexes, shp_v, color='red')
    ax[i, 1].scatter([0, 1, 2, 3, 4, 5, 6, 7, 8], shp, color='red')
    ax[i, 1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax[i, 1].set_xticklabels([j*dil for j in range(9)])
"""