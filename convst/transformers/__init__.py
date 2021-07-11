"""This module contains sklearn compatible transformers. Note that in ConvolutionalShapeletTransformer we do not make use
of the kernel or shapelet transformers to be able to use numba and other optimizations. Those transformers are mostly used
in examples or plotting scripts to facilitate the comprehension.
"""
from .convolutional_kernels import kernel, Rocket_kernel, MiniRocket_kernel
from .minirocket import MiniRocket
from .forest_splitter import ForestSplitter
from .convolutional_ST import ConvolutionalShapeletTransformer
from .shapelets import Convolutional_shapelet
from .convolutional_ST2 import ConvolutionalShapeletTransformer_onlyleaves
__author__ = 'Antoine Guillaume antoine.guillaume45@gmail.com'

__all__ = [ "MiniRocket","Convolutional_shapelet", "kernel",
 "Rocket_kernel", "MiniRocket_kernel", "ConvolutionalShapeletTransformer",
 "ConvolutionalShapeletTransformer_onlyleaves", "ForestSplitter"]