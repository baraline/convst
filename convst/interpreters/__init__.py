"""
This module contains interpreter objects, with an interpreter being linked to one algorithm. The goal of interperters is to provide
objects that can help build intuition in the decision process of a learned model, with a unified interface across the interpreters.
"""
__author__ = 'Antoine Guillaume antoine.guillaume45@gmail.com'

from .interpreter_cst import CST_interpreter

__all__ = ['CST_interpreter']