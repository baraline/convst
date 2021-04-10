# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:46:14 2021

@author: A694772
"""
__all__ = [
	"load_sktime_dataset_split"
]

from sktime.datasets.base import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder
from sktime.utils.data_processing import from_nested_to_3d_numpy


def load_sktime_dataset_split(name, normalize=True):
	#Load datasets
	X_train, y_train = load_UCR_UEA_dataset(name,return_X_y=True,split='train')
	X_test, y_test = load_UCR_UEA_dataset(name,return_X_y=True,split='test')
	
	#Convert pandas DataFrames to numpy arrays
	X_train = from_nested_to_3d_numpy(X_train)
	X_test = from_nested_to_3d_numpy(X_test)

	#Convert class labels to make sure they are between 0,n_classes
	le = LabelEncoder().fit(y_train)
	y_train = le.transform(y_train)
	y_test = le.transform(y_test)
	
	#Z-Normalize the data
	if normalize:            
		X_train = (X_train - X_train.mean(axis=-1, keepdims=True)) / (
			X_train.std(axis=-1, keepdims=True) + 1e-8)
		X_test = (X_test - X_test.mean(axis=-1, keepdims=True)) / (
			X_test.std(axis=-1, keepdims=True) + 1e-8)
	
	return X_train, X_test, y_train, y_test, le
	
	
def load_sktime_dataset(name, normalize=True):
	#Load datasets
	X, y = load_UCR_UEA_dataset(name,return_X_y=True)
	
	#Convert pandas DataFrames to numpy arrays
	X = from_nested_to_3d_numpy(X)

	#Convert class labels to make sure they are between 0,n_classes
	le = LabelEncoder().fit(y)
	y = le.transform(y)
	
	#Z-Normalize the data
	if normalize:            
		X = (X - X.mean(axis=-1, keepdims=True)) / (
			X.std(axis=-1, keepdims=True) + 1e-8)
		
	
	return X, y, le
	
