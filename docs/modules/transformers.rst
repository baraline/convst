.. _transformers:

=================================
Transformers availables in convst
=================================

.. currentmodule:: convst.transformers

The following sections will describe the main transformers that are available in the :mod:`convst.transformers` module.

Convolutional Shapelet Transform (CST)
--------------------------------------

The Convolutional Shapelet Transform (CST) algorithm, implemented by :class:`ConvolutionalShapeletTransformer` is 
an adaptation of time series shapelets which introduces in shapelets the notion of dilation used in convolutional kernels.

The goal of this algorithm is to extract discriminative patterns from the input space by leveraging the differences 
found in convolutional spaces, and build convolutional shapelets that will compare time series based on those extracted patterns.
For a detailed description on this algorithm, we refer the reader to [1]_ .

Add images with examples on GunPoint to show dilation, shapelet transform, convolutional and input space transistions ...


References
----------
.. [1] Antoine Guillaume et al, “Convolutional Shapelet Transform: A new approach of time series shapelets” (2021)


