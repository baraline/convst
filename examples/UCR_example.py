# -*- coding: utf-8 -*-

"""
Minimal example of RDST with visualisation
==========================================

This example give the minimal setup to run RDST and visualize the result
 on a dataset from the UCR archive.
"""

from convst.classifiers import R_DST_Ridge
from convst.utils.dataset_utils import load_sktime_dataset_split

# %%
# Load the dataset and run RDST with Ridge
# ----------------------------------------
#
# We load a UCR dataset with its name, and initialise RDST 
# with a Ridge classifier using the wrapper class R_DST_Ridge.

X_train, X_test, y_train, y_test, _ = load_sktime_dataset_split(
    'GunPoint', normalize=True)

rdst = R_DST_Ridge()

rdst.fit(X_train, y_train)
acc_score = rdst.score(X_test, y_test)

print("Accuracy Score for RDST : {}".format(acc_score))

# %%
# Visualize a shapelet 
# --------------------
#
# To visualize a shapelet, we use the associated function in the RDST class.
# We use the coefficients from the Ridge classifier to select a shapelet
# that was important for the classification task for one class. 

i_class = 0
ix = rdst.classifier['ridgeclassifiercv'].coef_[i_class].argsort()[0]
rdst.transformer.visualise_one_shapelet(ix//3, X_test, y_test, i_class, figsize=(17,12))
