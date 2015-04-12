# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:26:03 2015

@author: Diogo Silva


CUDA not implemented on this version.
"""

import numpy as np

import sys, traceback


class K_Means:       
       
    
    def __init__(self,N=None,D=None,K=None):
        self.N = N
        self.D = D
        self.K = K
    
    
    def fit(self, data, K, iters=3, cuda=False):
        
        N,D = data.shape
            
        self.N = N
        self.D = D
        self.K = K
        
        self.centroids = self._init_centroids(data)
        
        if iters == 0:
            return
        
        for i in xrange(iters):
            dist_mat = self._calc_dists(data,self.centroids,cuda=cuda)
            assign,grouped_data = self._assign_data(data,dist_mat)
            self.centroids =  self._np_recompute_centroids(grouped_data)

    def _init_centroids(self,data):
        
        centroids = np.empty((self.K,self.D),dtype=data.dtype)
        random_init = np.random.randint(0,self.N,self.K)
        self.init_seed = random_init
        
        for k in xrange(self.K):
            centroids[k] = data[random_init[k]]
        
        self.centroids = centroids
        
        return centroids

    def _calc_dists(self,data,centroids,cuda=False):
        if cuda:
            dist_mat = self._cu_calc_dists(data,centroids,gridDim=None,
                                           blockDim=None)#,memManage='manual')
        else:
            dist_mat = self._np_calc_dists(data,centroids)
            
        return dist_mat
            
    def _py_calc_dists(data,centroids):
        N,D = data.shape
        K,cD = centroids.shape

        for n in range(N):
            for k in range(K):
                dist=0
                for d in range(dim):
                    diff = a[n,d]-b[k,d]
                    dist += diff ** 2
                c[n,k]=dist
            
    def _np_calc_dists(self,data,centroids):
        """
        NumPy implementation - much faster than vanilla Python
        """
        N,D = data.shape
        K,cD = centroids.shape

        dist_mat = np.empty((N,K),dtype=data.dtype)    
        
        for k in xrange(K):
            dist = data - centroids[k]
            dist = dist ** 2
            dist_mat[:,k] = dist.sum(axis=1)
            
        return dist_mat
    
        
    def _assign_data(self,data,dist_mat):
        
        N,K = dist_mat.shape
        
        assign = np.argmin(dist_mat,axis=1)
        
        grouped_data=[[] for i in xrange(K)]
        
        
        for n in xrange(N):
            # add datum i to its assigned cluster assign[i]
            grouped_data[assign[n]].append(data[n])
        
        for k in xrange(K):
            grouped_data[k] = np.array(grouped_data[k])
        
        return assign,grouped_data
    
    def _np_recompute_centroids(self,grouped_data):
        
        # change to get dimension from class or search a non-empty cluster
        #dim = grouped_data[0][0].shape[1]
        dim = self.D
        K = len(grouped_data)
        
        centroids = np.empty((K,dim),dtype=grouped_data[0].dtype)
        
        for k in xrange(K):
            centroids[k] = np.mean(grouped_data[k],axis=0)
        
        return centroids