# -*- coding: utf-8 -*-

"""
Minimal example of CST with Interpreter
=========================

This example give the minimal setup to run CST and an interpreter
 on a dataset from the UCR archive.
"""

import numpy as np

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from convst.interpreters import CST_interpreter
from convst.transformers import ConvolutionalShapeletTransformer
from convst.utils import load_sktime_dataset_split

# %%
# Load the dataset and run CST with Ridge.
# ------------------------
#
# We load a UCR dataset with its name, and initialise CST 
# with a Ridge classifier. We then use CST to transform the inputs
# and feed it to the Ridge classifier

X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'GunPoint', normalize=True)

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

# %%
# Run the interpreter on a test sample
# ------------------------
#
# To run the interpreter on a sample, we use the `interpret_sample`
# function which take as input a 3D array, here of shape (1,1,n_timestamps)

icst = CST_interpreter(cst, X_train, X_cst_train, y_train)

i_sample=0
icst.interpret_sample(X_test[i_sample:i_sample+1])
