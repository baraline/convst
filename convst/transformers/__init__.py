"""Time series transformations.
"""
from .convolutional_kernels import kernel, Rocket_kernel, MiniRocket_kernel
from .minirocket import MiniRocket
from .convolutional_ST import ConvolutionalShapeletTransformer
from .shapelets import Convolutional_shapelet

__author__ = 'Antoine Guillaume antoine.guillaume45@gmail.com'

__all__ = [ "MiniRocket","Convolutional_shapelet", "kernel",
 "Rocket_kernel", "MiniRocket_kernel", "ConvolutionalShapeletTransformer"]