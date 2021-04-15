# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:51:20 2021

@author: A694772
"""
import numpy as np
from matplotlib import pyplot as plt
def generate_strides_1D(ts, window, dilation):
    """
    Generate strides from the input univariate time series with specified 
    length and dilation parameters.

    Parameters
    ----------
    ts : array, shape = (n_timestamps)
        An univariate time series, in a 1 dimensional view.
    window : int
        Length of the strides to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_samples, n_strides, stride_len)
        All possible subsequences of length stride_len for each time series.

    """
    shape = (ts.size - ((window-1)*dilation), window)
    strides = np.array([ts.strides[0], ts.strides[0]*dilation])
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

idx = np.zeros(100)
convs = [2,3,5,7,11]

for c in convs:
    pad = np.floor_divide((9 - 1) * c, 2)
    x = generate_strides_1D(np.array(range(0,100+2*pad)), 9, c) - pad
    for i in x:
        for j in i:
            if j>=0 and j<100:
                idx[j] += 1
                
idx1 = (idx - idx.min())/(idx.max() - idx.min())
plt.plot(idx1,label='prime')

idx = np.zeros(100)
convs = [1,2,3,4,5,6,7,8,9,10,11]

for c in convs:
    pad = np.floor_divide((9 - 1) * c, 2)
    x = generate_strides_1D(np.array(range(0,100+2*pad)), 9, c) - pad
    for i in x:
        for j in i:
            if j>=0 and j<100:
                idx[j] += 1
                
idx2 = (idx - idx.min())/(idx.max() - idx.min())
plt.plot(idx2,label='all')

idx = np.zeros(100)
convs = [1,3,5,7,9,11]

for c in convs:
    pad = np.floor_divide((9 - 1) * c, 2)
    x = generate_strides_1D(np.array(range(0,100+2*pad)), 9, c) - pad
    for i in x:
        for j in i:
            if j>=0 and j<100:
                idx[j] += 1
                
idx3 = (idx - idx.min())/(idx.max() - idx.min())
plt.plot(idx3,label='1/2')

idx = np.zeros(100)
convs = [1,4,7,10]

for c in convs:
    pad = np.floor_divide((9 - 1) * c, 2)
    x = generate_strides_1D(np.array(range(0,100+2*pad)), 9, c) - pad
    for i in x:
        for j in i:
            if j>=0 and j<100:
                idx[j] += 1
                
idx4 = (idx - idx.min())/(idx.max() - idx.min())
plt.plot(idx4,label='1/3')

plt.legend()
plt.show()
def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
    sieve[0] = False
    for i in range(int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)//3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]

p100 = []
all100 = []
half100 = []
third100 = []
a100 = []
b100 = []
for n in [10,100,1000,10000]:
    
    p100.append(primesfrom2to(n).size)
    all100.append(len(list(range(1,n))))
    half100.append(len(list(range(1,n,2))))
    third100.append(len(list(range(1,n,3))))
    a100.append(len(list(range(1,n,4))))
    b100.append(len(list(range(1,n,5))))
    
plt.plot(p100,label='prime')
plt.plot(all100,label='all')
plt.plot(half100,label='1/2')
plt.plot(third100,label='1/3')
plt.plot(a100,label='1/4')
plt.plot(b100,label='1/5')
plt.legend()
plt.show()