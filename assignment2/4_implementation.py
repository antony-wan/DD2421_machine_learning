#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:57:28 2019

@author: antony
"""

import numpy , random , math
from scipy.optimize import minimize 
import matplotlib.pyplot as plt

numpy.random.seed(100)
N = 10;
bounds=[(0, None) for b in range(N)];
start = numpy.zeros(N);

classA = numpy.concatenate ( (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
                               numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5])) ;
classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]
inputs = numpy.concatenate (( classA , classB )) 
targets = numpy.concatenate ((numpy.ones(classA.shape[0]) , 
                              -numpy.ones ( classB.shape[0] )))
N = inputs.shape[0] # Number of rows (samples)

permute=list(range(N)) 
random.shuffle( permute ) 
inputs = inputs[ permute, :]
targets = targets[ permute ]