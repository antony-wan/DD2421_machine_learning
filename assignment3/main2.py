#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:38:35 2019

@author: antony
"""

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random


# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.

X, labels = genBlobs()
# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    for jdx,k in enumerate(classes):
        idx = np.where(labels==k)[0]
        prior[jdx]=float(idx.shape[0])/float(Npts)
    # ==========================

    return prior

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))


    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    sigma_vector=np.zeros(Ndims)
    sigma_val=0
    
    for jdx,k in enumerate(classes):
        idx = np.where(labels==k)[0]
        xlc = X[idx,:] # Get the x for the class labels. Vectors are rows.
        wlc = W[idx,:]
        mu[jdx]=(np.sum(xlc*wlc, axis=0))/np.sum(wlc, axis=0)
               
        
        for m in range(Ndims):
            for i in range(xlc.shape[0]):
                sigma_val += (wlc[i]*(xlc[i][m]-mu[jdx][m])**2)/np.sum(wlc, axis=0)       
            sigma_vector[m] = sigma_val
            sigma_val=0
        
        sigma[jdx]=np.diag(sigma_vector)
        
    # ==========================

    return mu, sigma
    











