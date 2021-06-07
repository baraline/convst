from convst.utils import generate_strides_2D, generate_strides_1D
import numpy as np
import pytest

def init_numpy(dims):
    return np.random.random_sample(dims)
    
##########################################
#                                        #
#             Test strides               #
#                                        #
##########################################

@pytest.mark.parametrize("dims, window_size, dilation", [
    ((3, 100), 10, 1),
    ((3, 100), 5, 5),
    ((3, 100), 3, 15)
])
def test_strides_2D(dims, window_size, dilation):
    X = init_numpy(dims)
    X2 = generate_strides_2D(X, window_size, dilation)
    assert X2.shape == (dims[0], X.shape[1] - (window_size-1)*dilation, window_size)
    assert np.array_equal(X2[0,0], X[0,[0 + j*dilation for j in range(window_size)]])
   
    
@pytest.mark.parametrize("dims, window_size, dilation", [
    ((100), 10, 1),
    ((100), 5, 5),
    ((100), 3, 15)
])
def test_strides_1D(dims, window_size, dilation):
    X = init_numpy(dims)
    X2 = generate_strides_1D(X, window_size, dilation)
    assert X2.shape == (X.shape[0] - (window_size-1)*dilation, window_size)
    assert np.array_equal(X2[0], X[[0 + j*dilation for j in range(window_size)]])
    
