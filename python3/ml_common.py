#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ladd
# Date: 2017-09-03
import numpy as np

def softmax(inputs):    
    #Compute the softmax nonlinear activation function.
    exp_lists = np.e ** inputs
    rows= exp_lists.shape[0]
    sum = np.sum(exp_lists)
    for i in range(rows):
        exp_lists[i] = exp_lists[i]/sum
    return exp_lists

def ReLU(inputs):
    """
    Compute the ReLU: max(x, 0) nonlinearity.
    """
    outputs = inputs.copy()
    print("-执行前-ReLU---{}--".format(outputs))
    outputs[outputs<0] = 0
    print("-执行后-ReLU---{}--".format(outputs))
    return outputs

np.random.seed(42)
inputs = np.random.randint(1,8,4)
print(inputs-5)
print(softmax(inputs))
ReLU(inputs-7)