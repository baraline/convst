.. _introduction:

============
Introduction
============
The user guide in a work in progress.

Introduction
------------

A time series is a sequence of values indexed in chronological order. Time series data 
occurs in many different domains, in which it is often as important to efficiently 
solve the problem as to understand the reasoning behind the decision, 
to ensure that algorithms can be safely used on the field.

Notations
---------

Through the documentation of in the source code, we will use the notations detailed hereafter.

We represent a time series dataset as a three-dimensional array with shape (n_samples, n_features, n_timestamps), 
where the first axis represents the samples, the second axis represents a feature which is recorded in function of the third axis which represents time. 
In this context, an univariate time series dataset will be a array of shape (n_samples, 1, n_timestamps).

The input data will often be represented with a variable named X, X_train, or X_test if a model validation is being performed.

The class labels are represented as a one-dimensional array of shape (n_samples) indepentendly of the number of features or timestamps.
They are represented using a variable name such as y, y_train, or y_test if a model validation is being performed.