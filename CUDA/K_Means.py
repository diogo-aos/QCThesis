# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 08:53:20 2015

@author: Diogo Silva
"""

import numpy as np
import numbapro
from numbapro import *

class K_Means:       
       
    
    def __init__(self,N=None,D=None,K=None):
        self.N = N
        self.D = D
        self.K = K
        
        self._cudaDataRef = None
        
        self.__cuda = True
        self.__cuda_mem = "auto"

    @property    
    def cuda_mem(self):
        return self.__cuda_mem

    @cuda_mem.setter
    def cuda_mem(self,cuda_mem):
        if cuda_mem not in ['manual','auto']:
            raise Exception("cuda_mem = \'manual\' or \'auto\'")
    
    def fit(self,data,K,iters=3,cuda=True):
 
        if iters == 0:
            return
       
        N,D = data.shape
            
        self.N = N
        self.D = D
        self.K = K
        
        centroids = self._init_centroids(data)
        
        for i in xrange(iters):
            dist_mat = self._calc_dists(data,centroids,cuda=cuda)
            assign,grouped_data = self._assign_data(data,dist_mat)
            centroids =  self._np_recompute_centroids(grouped_data)
            self.centroids = centroids

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
                                           blockDim=None,memManage='manual')
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
    
    def _cu_calc_dists(self,data,centroids,gridDim=None,blockDim=None,
                       memManage='auto',keepDataRef=True):
        """
        TODO:
            - deal with gigantic data / distance matrix
            - deal with heavely assymetric distance matrix
                - if the number of blocks on any given dimension of 
                the grid > 2**16, divide that dimension by another dimension
                - don't forget to change the index computation in the kernel
        """
        
        
        N,D = data.shape
        K,cD = centroids.shape
        
        self.cuda_mem = memManage
        
        if self.__cuda_mem  not in ('manual','auto'):
            raise Exception("Invalid value for \'memManage\'.")

            
        if gridDim is None or blockDim is None:
            #dists shape
            

            MAX_THREADS_BLOCK = 16 * 20 # GT520M has 48 CUDA cores
            MAX_GRID_XYZ_DIM = 65535

            if K <= 28:
                blockWidth = K
                blockHeight = np.floor(MAX_THREADS_BLOCK / blockWidth)
                blockHeight = np.int(blockHeight)
            else:
                blockWidth = 20
                blockHeight = 16

            # grid width/height is the number of blocks necessary to fill
            # the columns/rows of the matrix
            gridWidth = np.ceil(np.float(K) / blockWidth)
            gridHeight = np.ceil(np.float(N) / blockHeight)

    
            blockDim = blockWidth, blockHeight
            gridDim = np.int(gridWidth), np.int(gridHeight)
        
        self.blockDim = blockDim
        self.gridDim = gridDim        
        
        distShape =  N,K
        dist_mat = np.empty(distShape,dtype=data.dtype)
        
        if self.__cuda_mem == 'manual':
            
            if keepDataRef:
                if self._cudaDataRef is None:
                    dData = cuda.to_device(data)
                    self._cudaDataRef = dData
                else:
                    dData = self._cudaDataRef
            else:
                dData = cuda.to_device(data)
                
            dCentroids = cuda.to_device(centroids)
            dDists = numbapro.cuda.device_array_like(dist_mat)
            
            self._cu_dist_kernel[gridDim,blockDim](dData,dCentroids,dDists)        
        
            dDists.copy_to_host(ary=dist_mat)
            numbapro.cuda.synchronize()

        elif self.__cuda_mem == 'auto':
            self._cu_dist_kernel[gridDim,blockDim](data,centroids,dist_mat) 
        
        return dist_mat
        
    
    @numbapro.cuda.jit("void(float32[:,:], float32[:,:], float32[:,:])")
    def _cu_dist_kernel(a,b,c):
        k,n = numbapro.cuda.grid(2)

        ch, cw = c.shape # c width and height

        if n >= ch or k >= cw:
            return

        dist = 0.0
        for d in range(a.shape[1]):
            diff = a[n,d]-b[k,d]
            dist += diff ** 2
        c[n,k]= dist
    
        
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
        
        centroids = np.empty((K,dim))
        for k in xrange(K):
            centroids[k] = np.mean(grouped_data[k],axis=0)
        
        return centroids


    def _cu_mean(self):

        pass