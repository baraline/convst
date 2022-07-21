#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:41:32 2022

@author: lifo
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

class c_StandardScaler(StandardScaler):
    def fit(self, X, y=None):
        self.usefull_atts = np.where(np.std(X, axis=0) != 0)[0]
        return super().fit(X[:, self.usefull_atts], y=y)
    
    def transform(self, X):
        return super().transform(X[:, self.usefull_atts])