# -*- coding: utf-8 -*-

"""
Example of changing numba configuration for RDST
================================================

This example shows how to modify the numba options to change the behaviour of 
numba functions
"""
# %%
# Load the module and change the numba options
# --------------------------------------------
#
# The __init__.py file of convst contains four variable that will change how 
# numba functions are compiled, please refer to the numba documentation for
# the effect of each option. In a classic desktop use of convst, you should
# not need to change any option. Issue have been known to occur on some HPC
# cluster. By default, the variables are all defined as True, here we change
# the value of __USE_NUMBA_CACHE__ and __USE_NUMBA_PARALLEL__ to False.

import convst

convst.__USE_NUMBA_CACHE__= False
convst.__USE_NUMBA_FASTMATH__ = True
convst.__USE_NUMBA_NOGIL__ = True
convst.__USE_NUMBA_PARALLEL__ = False

# %%
# Run convst with the modified numba options
# ------------------------------------------
#
# We can now use convst with these options, to check if the changes have been
# taken into account, we can inspect a numba function argument.
# !! THE MODIFICATION OF THE NUMBA CONFIGURATION MUST BE MADE BEFORE CALLING
# ANY NUMBA FUNCTION, NUMBA TREAT GLOBAL VARIABLE AT CONSTANT AT COMPILE TIME
# CHANGING THE VALUE OF THESE PARAMETER AFTER A FUNCTION IS COMPILED WILL NOT
# CHANGE ITS BEHAVIOR.

from convst.transformers._univariate_same_length import U_SL_apply_all_shapelets
print(U_SL_apply_all_shapelets.targetoptions)
