"""This module offers sklearn compatible classifiers which direclty wrap R_DST with some classifiers. They are equivalent to making a pipeline but are shorter to write. 
"""
from .rdst_ridge import R_DST_Ridge
__author__ = 'Antoine Guillaume antoine.guillaume45@gmail.com'

__all__ = ["R_DST_Ridge"]