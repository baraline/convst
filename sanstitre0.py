# -*- coding: utf-8 -*-

"""
Minimal example of CST with Interpreter
=======================================

This example give the minimal setup to run CST and an interpreter
 on a dataset from the UCR archive.
"""

from xgboost import XGBClassifier

from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer

from sklearn.pipeline import make_pipeline

import numpy as np

from sklearn.linear_model import RidgeClassifierCV

from convst.utils import load_sktime_dataset_split, stratified_resample

# %%
# Load the dataset and run CST with Ridge
# ---------------------------------------
#
# We load a UCR dataset with its name, and initialise CST
# with a Ridge classifier. We then use CST to transform the inputs
# and feed it to the Ridge classifier

X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'Adiac', normalize=True)


cst = ConvolutionalShapeletTransformer_onlyleaves(verbose=1,random_state=42,
                                                  leaves_only=True)

from sklearn.ensemble import RandomForestClassifier
rdg = RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                        normalize=True, class_weight='balanced')

X_cst_train = cst.fit_transform(X_train, y_train)
X_cst_test = cst.transform(X_test)
# In[]:
rdg = RandomForestClassifier(class_weight='balanced',max_features=0.25,ccp_alpha=0.02)
rdg.fit(X_cst_train, y_train)
pred = rdg.predict(X_cst_test)
print("F1 Score for CST : {}".format(
    f1_score(y_test, pred, average='macro')))
print('----------------\n')


# %%
# Run the interpreter on a test sample
# ------------------------------------
#
# To run the interpreter on a sample, we use the `interpret_sample`
# function which take as input a 3D array, here of shape (1,1,n_timestamps)


def _init_dataset(n_samples, n_timestamps, n_classes):
    X = np.zeros((n_samples, 1, n_timestamps))
    y = np.zeros(n_samples)
    n_sample_per_class = n_samples//n_classes
    r = n_samples % n_classes
    for i in range(n_classes):
        y[i*n_sample_per_class:(i+1)*n_sample_per_class] = i
    for i in range(r):
        y[-i] = i
    return X, y.astype(int)


def make_same_timestamps_diff_values(n_samples=100, n_timestamps=100, n_locs=3,
                                     n_classes=4, scale_diff=1, noise_coef=0.25,
                                     shape_coef=0.5):
    """
    This function generate a random dataset in which classes are discriminated only by value
    at the same specific locations for all classes.
    """
    X, y = _init_dataset(n_samples, n_timestamps, n_classes)
    base_data = np.random.rand(n_timestamps)
    locs = np.random.choice(range(n_timestamps), n_locs, replace=False)
    base_values = np.random.uniform(
        low=shape_coef, high=shape_coef*5, size=(n_locs))
    for i in range(n_samples):
        noise = np.random.normal(0, noise_coef, n_timestamps)
        X[i, 0] = base_data + noise
        X[i, 0, locs] += (base_values)*((1+y[i])*scale_diff)
    return X, y


pipe_rkt = make_pipeline(ConvolutionalShapeletTransformer_onlyleaves(),
                         XGBClassifier())

cv = cross_validate(pipe_rkt, X, y, cv=5, scoring={
                    'f1': make_scorer(f1_score, average='macro')}, n_jobs=None)
print(str(np.mean(cv['test_f1']))[0:5] +
      '(+/- ' + str(np.std(cv['test_f1']))[0:5] + ')')
