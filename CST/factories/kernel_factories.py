# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:46:14 2021

@author: A694772
"""
import numpy as np
from CST.base_transformers.convolutional_kernels import Rocket_kernel, Rocket_feature_kernel

class Rocket_factory():
    def __init__(self, rocket_object):
        self.rocket_object = rocket_object
        self.lengths = rocket_object.length_
        self.biases = rocket_object.bias_
        self.dilations =  rocket_object.dilation_
        self.paddings = rocket_object.padding_
        self.weights = rocket_object.weights_

    def create_feature_kernel(self, feature_id, id_ft=0):
        id_kernel = feature_id//2
        return Rocket_feature_kernel(length=self.lengths[id_kernel],
                                     bias=self.biases[id_kernel],
                                     dilation=self.dilations[id_kernel],
                                     padding=self.paddings[id_kernel],
                                     weights=self.weights[id_kernel, :self.lengths[id_kernel]], 
                                     id_ft=id_ft,
                                     feature_id=feature_id)
    
    def create_kernel(self, kernel_id, kernel_verbose=0, id_ft=0):
        return Rocket_kernel(length=self.lengths[kernel_id],
                             bias=self.biases[kernel_id],
                             dilation=self.dilations[kernel_id],
                             padding=self.paddings[kernel_id],
                             weights=self.weights[kernel_id, :self.lengths[kernel_id]], 
                             kernel_id=kernel_id,
                             id_ft=id_ft,
                             verbose=kernel_verbose)


    @property            
    def rocket_object(self):
        return self._rocket_object
      
    @rocket_object.setter   
    def rocket_object(self, value):
        self._rocket_object = value
        
    @property            
    def lengths(self):
        return self._lengths
      
    @lengths.setter   
    def lengths(self, value):
        self._lengths = value
        
    @property            
    def biases(self):
        return self._biases
      
    @biases.setter   
    def biases(self, value):
        self._biases = value
        
    @property            
    def dilations(self):
        return self._dilations
      
    @dilations.setter   
    def dilations(self, value):
        self._dilations = value
        
    @property            
    def paddings(self):
        return self._paddings
      
    @paddings.setter   
    def paddings(self, value):
        self._paddings = value
        
    @property            
    def weights(self):
        return self._weights
      
    @weights.setter   
    def weights(self, value):
        self._weights = value
        
        
class Rocket_factory_sktime():
    def __init__(self, rocket_object):
        self.rocket_object = rocket_object
        self.lengths = rocket_object.kernels[1]
        self.biases = rocket_object.kernels[2]
        self.dilations = rocket_object.kernels[3]
        self.paddings = rocket_object.kernels[4]
        a=0
        b=0
        weights = []
        for i in range(rocket_object.num_kernels):
            b+= rocket_object.kernels[1][i]
            weights.append(np.asarray(rocket_object.kernels[0][a:b]))
            a+= rocket_object.kernels[1][i]
        self._weights = np.asarray(weights,dtype='object')
    
    def create_feature_kernel(self, feature_id):
        id_kernel = feature_id//2
        return Rocket_feature_kernel(length=self.lengths[id_kernel],
                                     bias=self.biases[id_kernel],
                                     dilation=self.dilations[id_kernel],
                                     padding=self.paddings[id_kernel],
                                     weights=self.weights[id_kernel], 
                                     feature_id=feature_id)
    
    def create_kernel(self, id_kernel, kernel_verbose=0):
        return Rocket_kernel(length=self.lengths[id_kernel],
                             bias=self.biases[id_kernel],
                             dilation=self.dilations[id_kernel],
                             padding=self.paddings[id_kernel],
                             weights=self.weights[id_kernel], 
                             rocket_id=id_kernel,
                             verbose=kernel_verbose)
    
    @property            
    def rocket_object(self):
        return self._rocket_object
      
    @rocket_object.setter   
    def rocket_object(self, value):
        self._rocket_object = value
        
    @property            
    def lengths(self):
        return self._lengths
      
    @lengths.setter   
    def lengths(self, value):
        self._lengths = value
        
    @property            
    def biases(self):
        return self._biases
      
    @biases.setter   
    def biases(self, value):
        self._biases = value
        
    @property            
    def dilations(self):
        return self._dilations
      
    @dilations.setter   
    def dilations(self, value):
        self._dilations = value
        
    @property            
    def paddings(self):
        return self._paddings
      
    @paddings.setter   
    def paddings(self, value):
        self._paddings = value
        
    @property            
    def weights(self):
        return self._weights
      
    @weights.setter   
    def weights(self, value):
        self._weights = value
    