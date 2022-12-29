# -*- coding: utf-8 -*-
"""
@author: Antoine Guillaume
"""
from numpy.random import choice, uniform, random, seed
from numpy import (
    unique, where, percentile, all as _all, int64, bool_,
    log2, floor_divide, zeros, floor, power, ones, cumsum, mean, std
)

from convst.transformers._commons import (
    get_subsequence, compute_shapelet_dist_vector,
    apply_one_shapelet_one_sample_univariate, _combinations_1d,
    generate_strides_1D, prime_up_to, choice_log
)

from numba import njit, prange

@njit(cache=True, nogil=True)
def U_SL_init_random_shapelet_params(
    n_shapelets, shapelet_sizes, n_timestamps, p_norm, prime_scheme
):
    """
    Initialize the parameters of the shapelets.    

    Parameters
    ----------
    n_shapelets : int
        Number of shapelet to initialize
    shapelet_sizes : array, shape=()
        Set of possible length for the shapelets
    n_timestamps : int
        Number of timestamps in the input data
    p_norm : float
        A value in the range [0,1] indicating the chance for each
        shapelet to use z-normalized distance

    Returns
    -------
    values : array, shape=(n_shapelet, max(shapelet_sizes))
        An initialized (empty) value array for each shapelet
    lengths : array, shape=(n_shapelet)
        The randomly initialized length of each shapelet
    dilations : array, shape=(n_shapelet)
        The randomly initialized dilation of each shapelet
    threshold : array, shape=(n_shapelet)
        An initialized (empty) value array for each shapelet
    normalize : array, shape=(n_shapelet)
        The randomly initialized normalization indicator of each shapelet

    """
    # Lengths of the shapelets
    lengths = choice(shapelet_sizes, size=n_shapelets).astype(int64)
    # Dilations
    upper_bounds = log2(floor_divide(n_timestamps - 1, lengths - 1))
    if prime_scheme:
        primes = prime_up_to(int64(2**upper_bounds.max()))
        dilations = zeros(n_shapelets, dtype=int64)
        #TODO : optimize to avoid recomputing choice log for all upper bounds
        #Loop on each unique upper bounds ?
        for i in prange(n_shapelets):
            shp_primes = primes[primes<=int64(2**upper_bounds[i])]
            dilations[i] = shp_primes[choice_log(shp_primes.shape[0], 1)[0]]
    else:
        powers = zeros(n_shapelets)
        for i in prange(n_shapelets):
            powers[i] = uniform(0, upper_bounds[i])
        dilations = floor(power(2, powers)).astype(int64)
        
    # Init threshold array
    threshold = zeros(n_shapelets)
    
    # Init values array
    values = zeros((n_shapelets, max(shapelet_sizes)))
    
    # Is shapelet using z-normalization ?
    normalize = random(size=n_shapelets)
    normalize = (normalize < p_norm)

    return values, lengths, dilations, threshold, normalize

@njit(cache=True, parallel=True, nogil=True)
def U_SL_generate_shapelet(
    X, y, n_shapelets, shapelet_sizes, r_seed, p_norm, p_min, p_max, alpha,
    dist_func, use_phase, prime_scheme
):
    """
    Given a time series dataset and parameters of the method, generate the
    set of random shapelet that will be used in the rest of the algorithm.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Time series dataset
    y : array, shape=(n_samples)
        Class of each input time series
    n_shapelets : int
        Number of shapelet to generate
    shapelet_sizes : array, shape=()
        An array of possible shapelet length.
    seed : int
        Random seed generator for numpy
    p_norm : float
        Probability of each shapelet to use z-normalized distance
    p_min : float
        Lower bound for the percentile during the choice of threshold
    p_max : float
        Upper bound for the percentile during the choice of threshold
    alpha : float
        Alpha similarity parameter
    dist_func: function
        A distance function implemented with Numba taking two 1D vectors as
        input.
    use_phase: bool
        Wheter to use phase invariance
    
    Returns
    -------
    set of array, shape=(5)
        values : array, shape=(n_shapelets, max(shapelet_sizes))
            Values of the shapelets. If the shapelet use z-normalized distance,
            those values are already z-normalized by the shapelet 
            initialization step.
        lengths : array, shape=(n_shapelets)
            Length parameter of the shapelets
        dilations : array, shape=(n_shapelets)
            Dilation parameter of the shapelets
        threshold : array, shape=(n_shapelets)
            Threshold parameter of the shapelets
        normalize : array, shape=(n_shapelets)
            Normalization indicatorr of the shapelets
    """
    n_samples, n_features, n_timestamps = X.shape
    # Fix the random seed
    seed(r_seed)

    #Initialize shapelets
    values, lengths, dilations, threshold, normalize = \
    U_SL_init_random_shapelet_params(
        n_shapelets, shapelet_sizes, n_timestamps, p_norm, prime_scheme
    )
    #Initialize self similarity mask
    unique_dil = unique(dilations)
    mask_sampling = ones(
        (2,unique_dil.shape[0],n_samples,n_timestamps), dtype=bool_
    )

    #For each dilation, we can do in parallel
    for i_d in prange(unique_dil.shape[0]):
        #For each shapelet id with this dilation
        id_shps = where(dilations==unique_dil[i_d])[0]
        min_l = min(lengths[id_shps])
        for i_shp in id_shps:
            _dilation = dilations[i_shp]
            _length = lengths[i_shp]
            norm = int64(normalize[i_shp])
            if use_phase:
                d_shape = n_timestamps
            else:
                d_shape = n_timestamps-(_length-1)*_dilation
            mask_dil = mask_sampling[norm,i_d]
            
            #Possible sampling points given self similarity mask
            i_mask = where(mask_dil[:,:d_shape])
            
            if i_mask[0].shape[0] > 0:
                #Choose a sample
                id_sample = choice(i_mask[0])
                #Choose a timestamp
                index = choice(i_mask[1][i_mask[0]==id_sample])
                #Update the mask
                alpha_size = _length - int64(max(1,(1-alpha)*min_l))
                for j in range(alpha_size):
                    #We can use modulo even without phase invariance, as we
                    #limit the sampling to d_shape
                    mask_sampling[
                        norm, i_d, id_sample,
                        (index-(j*_dilation))%n_timestamps
                    ] = False
                    mask_sampling[
                        norm, i_d, id_sample,
                        (index+(j*_dilation))%n_timestamps
                    ] = False
                
                #Extract the values
                v = get_subsequence(
                    X[id_sample, 0], index, _length, _dilation, norm, use_phase
                )
        
                #Select another sample of the same class as the sample used to
                loc_others = where(y == y[id_sample])[0]
                if loc_others.shape[0] > 1:
                    loc_others = loc_others[loc_others != id_sample]
                    id_test = choice(loc_others)
                else:
                    id_test = id_sample
                
                #Compute distance vector
                x_dist = compute_shapelet_dist_vector(
                    X[id_test, 0], v, _length, _dilation, dist_func, norm,
                    use_phase
                )
                
                #Extract value between two percentile as threshold for SO
                ps = percentile(x_dist, [p_min,p_max])
                threshold[i_shp] = uniform(
                    ps[0], ps[1]
                )
                values[i_shp, :_length] = v
                
    mask_values = ones(n_shapelets, dtype=bool_)
    for i in prange(n_shapelets):
        if _all(values[i] == 0):
            mask_values[i] = False
    
    return (
        values[mask_values],
        lengths[mask_values],
        dilations[mask_values],
        threshold[mask_values],
        normalize[mask_values]
    )


@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def U_SL_apply_all_shapelets(
    X, shapelets, dist_func, use_phase
):
    """
    Apply a set of generated shapelet using the parameter arrays previously 
    generated to a set of time series.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Input time series
    shapelets: set of array, shape=(5)
        values : array, shape=(n_shapelets, max(shapelet_sizes))
            Values of the shapelets. If the shapelet use z-normalized distance,
            those values are already z-normalized by the shapelet 
            initialization step.
        lengths : array, shape=(n_shapelets)
            Length parameter of the shapelets
        dilations : array, shape=(n_shapelets)
            Dilation parameter of the shapelets
        threshold : array, shape=(n_shapelets)
            Threshold parameter of the shapelets
        normalize : array, shape=(n_shapelets)
            Normalization indicatorr of the shapelets
    dist_func: function
        A distance function implemented with Numba taking two 1D vectors as
        input.
    use_phase: bool
        Wheter to use phase invariance
    
    Returns
    -------
    X_new : array, shape=(n_samples, 3*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        feature from the distance vector computed on each time series.

    """
    (values, lengths, dilations, threshold, normalize) = shapelets
    n_shapelets = len(lengths)
    n_samples, n_ft, n_timestamps = X.shape
    n_features = 3

    # Existing combination of length and dilations
    params_shp = _combinations_1d(lengths, dilations)
    n_shp_params = zeros(params_shp.shape[0]+1, dtype=int64)
    
    idx_shp = zeros(n_shapelets, dtype=int64)
    
    a = 0
    
    for i in range(params_shp.shape[0]):
        _length = params_shp[i, 0]
        _dilation = params_shp[i, 1]
        
        ix_shapelets = where((lengths == _length) & (dilations == _dilation))[0]
        b = a + ix_shapelets.shape[0]
        
        idx_shp[a:b] = ix_shapelets
        n_shp_params[i+1] = ix_shapelets.shape[0]
        
        a = b
    n_shp_params = cumsum(n_shp_params)
    
    X_new = zeros((n_samples, n_features * n_shapelets))
    for i_sample in prange(n_samples):
        #n_shp_params is a cumsum starting at 0
        for i_shp_param in prange(n_shp_params.shape[0]-1):
            _length = params_shp[i_shp_param, 0]
            _dilation = params_shp[i_shp_param, 1]
            
            strides = generate_strides_1D(
                X[i_sample, 0], _length, _dilation, use_phase
            )
            # Indexes of shapelets corresponding to the params of i_shp_param
            _idx_shp = idx_shp[n_shp_params[i_shp_param]:n_shp_params[i_shp_param+1]]
            
            _idx_no_norm = _idx_shp[where(normalize[_idx_shp] == False)[0]]
            for i_idx in range(_idx_no_norm.shape[0]):               
                i_shp = _idx_no_norm[i_idx]
                _values = values[i_shp, :_length]
                
                X_new[i_sample, (n_features * i_shp):(n_features * i_shp + n_features)] = \
                apply_one_shapelet_one_sample_univariate(
                    strides, _values, threshold[i_shp], dist_func
                )
            
            _idx_norm = _idx_shp[where(normalize[_idx_shp] == True)[0]]
            if _idx_norm.shape[0] > 0:
                for i_stride in range(strides.shape[0]):
                    _str = strides[i_stride]
                    strides[i_stride] = (_str - mean(_str))/(std(_str)+1e-8)
                        
                for i_idx in range(_idx_norm.shape[0]):               
                    i_shp = _idx_norm[i_idx]
                    _values = values[i_shp, :_length]
                    
                    X_new[i_sample, (n_features * i_shp):(n_features * i_shp + n_features)] = \
                    apply_one_shapelet_one_sample_univariate(
                        strides, _values, threshold[i_shp], dist_func
                    )
                
    return X_new