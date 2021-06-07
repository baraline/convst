# -*- coding: utf-8 -*-

"""
This is my example script
=========================

This example 
"""

import numpy as np

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from convst.interpreters import CST_interpreter
from convst.transformers import ConvolutionalShapeletTransformer
from convst.utils import load_sktime_dataset_split

# %%
# This is a section header
# ------------------------
#
# In the built documentation, it will be rendered as rST. All rST lines
# must begin with '# ' (note the space) including underlines below section
# headers.

# Load UCR Dataset by names

X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'GunPoint', normalize=True)

# First run will be slow due to numba compilations on the first call. Run small dataset like GunPoint the first time ! 
# Put verbose = 1 to activate the verbose progression of the algorithm.

cst = ConvolutionalShapeletTransformer(verbose=0)
rdg = RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), 
                        normalize=True, class_weight='balanced')

X_cst_train = cst.fit_transform(X_train, y_train)
X_cst_test = cst.transform(X_test)

rdg.fit(X_cst_train, y_train)
pred = rdg.predict(X_cst_test)
print("Accuracy Score for CST : {}".format(accuracy_score(y_test, pred)))
print("Accuracy Score for CST : {}".format(
    f1_score(y_test, pred, average='macro')))

icst = CST_interpreter(cst, X_train, X_cst_train, y_train)

i_sample=0
icst.interpret_sample(X_test[i_sample:i_sample+1])
