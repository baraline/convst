"""This module contains sklearn compatible transformers. Note that in ConvolutionalShapeletTransformer we do not make use
of the kernel or shapelet transformers to be able to use numba and other optimizations. Those transformers are mostly used
in examples or plotting scripts to facilitate the comprehension.
"""
from .convolutional_kernels import kernel, Rocket_kernel, MiniRocket_kernel
from .minirocket import MiniRocket
from .convolutional_ST import ConvolutionalShapeletTransformer
from .shapelets import Convolutional_shapelet

__author__ = 'Antoine Guillaume antoine.guillaume45@gmail.com'

__all__ = [ "MiniRocket","Convolutional_shapelet", "kernel",
 "Rocket_kernel", "MiniRocket_kernel", "ConvolutionalShapeletTransformer"]