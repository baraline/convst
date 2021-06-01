# -*- coding: utf-8 -*-

import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from CST.interpreters.interpreter_cst import interpeter_cst
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from CST.utils.dataset_utils import load_sktime_dataset_split

# Load Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'GunPoint', normalize=True)

# In[]:
# First run will be slow due to numba compilations on the first call. Run small dataset like GunPoint the first time !
# Put verbose = 1 to activate the verbose progression of the algorithm.

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

# In[]:
X_cst = cst[0].transform(X_train)
icst = interpeter_cst(cst[0], X_train, X_cst, y_train)

# In[]:
i_sample=0
icst.interpret_sample(X_test[i_sample:i_sample+1])