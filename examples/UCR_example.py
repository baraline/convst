# -*- coding: utf-8 -*-

"""
Minimal example of RDST with visualisation
==========================================

This example give the minimal setup to run RDST and visualize the result
 on a dataset from the UCR archive.
"""

# %%
# Load the dataset and run RDST with Ridge
# ----------------------------------------
#
# We load a UCR dataset with its name, and initialise RDST 
# with a Ridge classifier using the wrapper class R_DST_Ridge.

from convst.classifiers import R_DST_Ridge
from convst.utils.dataset_utils import load_sktime_dataset_split

X_train, X_test, y_train, y_test, _ = load_sktime_dataset_split(
    'GunPoint', normalize=True)

rdst = R_DST_Ridge(n_shapelets=10000, n_jobs=-1)

rdst.fit(X_train, y_train)
acc_score = rdst.score(X_test, y_test)

print("Accuracy Score for RDST : {}".format(acc_score))

# %%
# Visualize the best shapelet of a class 
# --------------------------------------
#
# To visualize a shapelet, we use the dedicated interpreter class.
# It uses the coefficients from the Ridge classifier to select a shapelet
# that was important for the classification task for one class. We can then
# plot the distribution of features the shapelet generate for both the training
# and the testing set, too see if the distributions are alike.  

from convst.interpreters import RDST_Ridge_interpreter
target_class = 1
interpreter = RDST_Ridge_interpreter(rdst)

interpreter.visualize_best_shapelets_one_class(
    X_train, y_train, target_class, n_shp=1
)

interpreter.visualize_best_shapelets_one_class(
    X_test, y_test, target_class, n_shp=1
)


