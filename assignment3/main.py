#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:42:04 2019

@author: antony
"""
import numpy as np

# Dimensions of Matrices
rows = 7
cols = 4
depth = 5
# Creating matrices
A = np.zeros((rows,cols)) # 2D Matrix of zeros
print(A.shape)
A = np.zeros((depth,rows,cols))  # 3D Matrix of zeros
A = np.ones((rows,cols)) # 2D Matrix of ones
A = np.array([(1,2,3),(4,5,6),(7,8,9)]) # 2D 3x3 matrix with values
# Turn it into a square diagonal matrix with zeros of-diagonal
d = np.diag(A) # Get diagonal as a row vector
d = np.diag(d) # Turn a row vector into a diagonal matrix