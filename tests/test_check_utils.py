from convst.utils import check_array_3D, check_array_2D, check_array_1D
import numpy as np
import pytest


def init_numpy(dims):
    return np.random.random_sample(dims)
    
##########################################
#                                        #
#             Test 3D checks             #
#                                        #
##########################################

@pytest.mark.parametrize("dims", [
    ((30, 15, 12)),
    ((2, 1, 18)),
])
def test_check_3D_numpy_no_change(dims):
    X = init_numpy(dims)
    X2 = check_array_3D(X)
    assert np.array_equal(X,X2)


@pytest.mark.parametrize("dims", [
    ((30, 15)),
    ((30)),
])
def test_check_3D_numpy_wrong_dimensions(dims):
    X = init_numpy(dims)
    with pytest.raises(ValueError):
        check_array_3D(X)
    
@pytest.mark.parametrize("dims", [
    ((30, 0, 15)),
    ((30, 15, 0)),
    ((0, 15, 30)),
])
def test_check_3D_numpy_empty_dimensions(dims):
    X = init_numpy(dims)
    with pytest.raises(ValueError):
        check_array_3D(X)

##########################################
#                                        #
#             Test 2D checks             #
#                                        #
##########################################  

@pytest.mark.parametrize("dims", [
    ((30, 15)),
    ((2, 1)),
])
def test_check_2D_numpy_no_change(dims):
    X = init_numpy(dims)
    X2 = check_array_2D(X)
    assert np.array_equal(X,X2)


@pytest.mark.parametrize("dims", [
    ((30, 15, 20)),
    ((30)),
])
def test_check_2D_numpy_wrong_dimensions(dims):
    X = init_numpy(dims)
    with pytest.raises(ValueError):
        check_array_2D(X)
    
@pytest.mark.parametrize("dims", [
    ((30, 0)),
    ((0, 15)),
])
def test_check_2D_numpy_empty_dimensions(dims):
    X = init_numpy(dims)
    with pytest.raises(ValueError):
        check_array_2D(X)
        
##########################################
#                                        #
#             Test 1D checks             #
#                                        #
##########################################        

@pytest.mark.parametrize("dims", [
    ((30)),
    ((2)),
])
def test_check_1D_numpy_no_change(dims):
    X = init_numpy(dims)
    X2 = check_array_1D(X)
    assert np.array_equal(X,X2)


@pytest.mark.parametrize("dims", [
    ((30, 15)),
    ((30, 15, 30)),
])
def test_check_1D_numpy_wrong_dimensions(dims):
    X = init_numpy(dims)
    with pytest.raises(ValueError):
        check_array_1D(X)


@pytest.mark.parametrize("dims", [
    ((0)),
])
def test_check_1D_numpy_empty_dimensions(dims):
    X = init_numpy(dims)
    with pytest.raises(ValueError):
        check_array_1D(X)
        

##########################################
#                                        #
#            Test type checks            #
#                                        #
##########################################  
    
@pytest.mark.parametrize("X", [
    ("3"),
    (1.0),
    (1),
    ((1,1,1)),
])
def test_type_check(X):
    with pytest.raises(ValueError):
        check_array_1D(X)
        
