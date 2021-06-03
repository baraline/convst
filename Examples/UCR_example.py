# -*- coding: utf-8 -*-

import numpy as np

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from convst.interpreters.interpreter_cst import interpeter_cst
from convst.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from convst.utils.dataset_utils import load_sktime_dataset_split

# Load Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'GunPoint', normalize=True)

# In[]:
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

# In[]:
icst = interpeter_cst(cst, X_train, X_cst_train, y_train)

i_sample=0
icst.interpret_sample(X_test[i_sample:i_sample+1])
