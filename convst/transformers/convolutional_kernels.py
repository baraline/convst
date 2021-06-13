# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from convst.utils.kernel_utils import apply_one_kernel_one_sample, apply_one_kernel_all_sample
from convst.utils.checks_utils import check_array_3D, check_array_1D


class kernel(BaseEstimator, TransformerMixin):
    """
    Basic implementation of a convolutional kernel

    Parameters
    ----------
    bias : float
        Bias of the kernel.
    dilation : int
        Dilation of the kernel. 
    padding : int
        Padding of the kernel.
    weights : array, shape = (length)
        Weights of the kernel.


    """

    def __init__(self, bias=None, dilation=None, padding=None,
                 weights=None, id_ft=0):

        self.bias = bias
        self.dilation = dilation
        self.padding = padding
        self.weights = weights
        self.length = self.weights.shape[0]

    def print_info(self):
        """
        Print the kernel parameters

        Returns
        -------
        None.

        """
        print('\n----------------- KERNEL INFO -----------------')
        print('-- LENGHT : {}'.format(self.length))
        print('-- BIAS : {}'.format(self.bias))
        print('-- DILATION : {}'.format(self.dilation))
        print('-- PADDING : {}'.format(self.padding))
        print('-- WEIGHTS : {}'.format(self.weights))

    def fit(self, X, y=None, normalise=True):
        """
        Placeholder method to allow fit_transform method.

        Parameters
        ----------
        X : ignored

        y : ignored, optional

        normalise : ignored, optional


        Returns
        -------
        self

        """
        self._check_is_init()
        return self

    def transform(self, X, normalise=True):
        """
        Apply the kernel through a convolution operation on each time series of the input.
        If the input is have multiple feature, it will apply the convolution to all of them.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features, n_timestamps)
            Input time series to convolve
        normalise : boolean, optional
            If True, X will be normalised before applying the convolution.
            The default is True.

        Returns
        -------
        array, shape = (n_samples, n_features, n_convolution)
            The convolved input time series.
        """

        self._check_is_init()
        X = check_array_3D(X, coerce_to_numpy=True)
        if normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )

        n_conv = X.shape[2] - ((self.length - 1) *
                               self.dilation) + (2 * self.padding)
        conv = np.zeros(X.shape[0], X.shape[1], n_conv)

        for i in range(X.shape[1]):
            conv[:, i] += apply_one_kernel_all_sample(X, i, self.weights,
                                                      self.length, self.bias,
                                                      self.dilation, self.padding)
        return conv

    def _convolve_one_sample(self, X):
        """
        Apply the kernel through a convolution operation to the input time series

        Parameters
        ----------
        X : array, shape = (n_timestamps)
            A univariate time series.

        Returns
        -------
        array, shape = (n_convolution)
            The result of the convolution.

        """
        X = check_array_1D(X)
        return apply_one_kernel_one_sample(X, X.shape[0], self.weights,
                                           self.length, self.bias,
                                           self.dilation, self.padding)

    def _get_indexes(self, convolution_index):
        """
        Return the indexes of the input values used by the convolution to
        generate the value at the convolution_index parameter

        Parameters
        ----------
        convolution_index : int
            An index of the convolution between input and kernel

        Returns
        -------
        array, shape = (length)
            An array which contain the indexes of the input values
            used by the convolution operation at convolution_index.

        """
        return np.asarray([convolution_index + (l*self.dilation) - self.padding
                           for l in range(self.length)])

    def _check_is_init(self):
        if any(self.__dict__[attribute] is None for attribute in ['_bias',
                                                                  '_dilation',
                                                                  '_padding',
                                                                  '_weights']):
            raise AttributeError("Kernel attribute not initialised correctly, "
                                 "at least one attribute was set to None")

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value

    @property
    def dilation(self):
        return self._dilation

    @dilation.setter
    def dilation(self, value):
        self._dilation = value

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        self._padding = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        value = check_array_1D(value)
        self._weights = value
        self.length = self.weights.shape[0]


class Rocket_kernel(kernel):
    """
    Wrapper for kernels extracted from ROCKET.
    Support multivariate in a basic way by convolving all features.

    Parameters
    ----------
    bias : float
        Bias of the kernel.
    dilation : int
        Dilation of the kernel.
    padding : int
        Padding of the kernel. 
    weights : array, shape = (length)
        Weights of the kernel. 


    """

    def __init__(self, bias=None, dilation=None, padding=None,
                 weights=None):

        super().__init__(bias=bias, dilation=dilation,
                         padding=padding, weights=weights)

    def fit(self, X, y=None):
        """
        Placeholder method to allow fit_transform method.

        Parameters
        ----------
        X : ignored

        y : ignored, optional

        normalise : ignored, optional


        Returns
        -------
        self

        """
        self._check_is_init()
        return self

    def transform(self, X, normalise=True, to_feature=True):
        """
        Apply the kernel through a convolution operation on each time series
        of the input. If the input is have multiple feature, it will apply the
        convolution to all of them.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features, n_timestamps)
            Input time series to convolve
        normalise : boolean, optional
            If True, X will be normalised before applying the convolution.
            The default is True.
         to_feature : boolean, optional
            If True, the function will return the ppv and max features
            of the convolution. The default is True.

        Returns
        -------
        array, shape = (n_samples, n_features, 2)
            The convolved input time series on which the ppv and max 
            features were extracted if to_feature is True. Else it return
            the raw convolutions.
        """
        self._check_is_init()
        conv = super().transform(X, normalise=normalise)

        if to_feature:
            features = np.zeros(X.shape[0], X.shape[1], 2)
            for i in range(X.shape[1]):
                features[:, i, 0] += self._ft_ppv(conv[:, i])
                features[:, i, 1] += self._ft_max(conv[:, i])
            return features
        else:
            return conv

    def _ft_max(self, X_conv):
        return np.max(X_conv, axis=-1)

    def _ft_ppv(self, X_conv):
        return np.mean(X_conv > 0, axis=-1)

    def _ft_max_loc(self, conv):
        return (conv == conv.max()).nonzero()[0]

    def _ft_ppv_loc(self, conv):
        return (conv > 0).nonzero()[0]


class MiniRocket_kernel(kernel):
    """
    Wrapper for kernels extracted from Mini-ROCKET.
    Support multivariate in a basic way by convolving all features.

    Parameters
    ----------
    bias : float
        Bias of the kernel.
    dilation : int
        Dilation of the kernel.
    padding : int
        Padding of the kernel.
    weights : array, shape = (length)
        Weights of the kernel.


    """

    def __init__(self, bias=None, dilation=None, padding=None,
                 weights=None):

        super().__init__(bias=bias, dilation=dilation,
                         padding=padding, weights=weights)

    def fit(self, X, y=None):
        """
        Placeholder method to allow fit_transform method.

        Parameters
        ----------
        X : ignored

        y : ignored, optional

        normalise : ignored, optional


        Returns
        -------
        self

        """
        self._check_is_init()
        return self

    def transform(self, X, normalise=True, to_feature=True):
        """
        Apply the kernel through a convolution operation on each time series of the input.
        If the input is have multiple feature, it will apply the convolution to all of them.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features, n_timestamps)
            Input time series to convolve
        normalise : boolean, optional
            If True, X will be normalised before applying the convolution.
            The default is True.
         to_feature : boolean, optional
            If True, the function will return the ppv feature
            of the convolution. The default is True.

        Returns
        -------
        array, shape = (n_samples, n_features)
            The convolved input time series on which the ppv feature was
            extracted if to_feature is True. Else it return the raw
            convolutions.
        """
        self._check_is_init()
        conv = super().transform(X, normalise=normalise)

        if to_feature:
            features = np.zeros(X.shape[0], X.shape[1])
            for i in range(X.shape[1]):
                features[:, i] += self._ft_ppv(conv[:, i])

            return features
        else:
            return conv

    def _ft_ppv(self, X_conv):
        return np.mean(X_conv > 0, axis=-1)

    def _ft_ppv_loc(self, conv):
        return (conv > 0).nonzero()[0]
