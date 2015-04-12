# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:54:02 2015

@author: Diogo Silva
"""

import numpy as np
import numbapro
from numbapro import *  

import sys, traceback


class K_Means:       
       
    
    def __init__(self,N=None,D=None,K=None):
        self.N = N
        self.D = D
        self.K = K
        
        self._converge = False       
        self._mode = "cuda"
        
        self._error = None

        self._cudaDataRef = None
        self._cuda = True
        self._cuda_mem = "auto"
        self._gridDim = None
        self._blockDim = None
        self._dist_kernel = 0 # 0 = normal index, 1 = special grid index


        # outputs
        self.inertia_ = None
        self._iters = 0
        
    """
    def get_cuda_mem(self):
        return self._cuda_mem

    def set_cuda_mem(self,newVal):
        print newVal
        if newVal not in ('manual','auto'):
            raise Exception("cuda_mem = \'manual\' or \'auto\'")
        else:
            self._cuda_mem = newVal
    """
    
    
    def fit(self, data, K, iters=3, mode="cuda", cuda_mem='manual',tol=1e-4,max_iters=300):
        
        N,D = data.shape
            
        self.N = N
        self.D = D
        self.K = K

        # TODO check parameters, check iters is number or "converge"

        self._mode = mode
        
        self.centroids = self._init_centroids(data)
        
        if iters == 0:
            return
        
        if iters == "converge":
            self._converge = True
            self._maxiters = max_iters
        else:
            self._maxiters = iters
        
        stopcond = False
        self._iters = 0

        while not stopcond:
            dist_mat = self._calc_dists(data,self.centroids)
            assign,grouped_data = self._assign_data(data,dist_mat)
            self.centroids =  self._np_recompute_centroids(grouped_data)

            self._iters += 1 #increment iteration counter

            # evaluate stop condition
            if self._converge:
                # stop if reached max iterations
                if self._iters >= self._maxiters:
                    stopcond = True
                # stop if convergence tolerance achieved
                elif self._error <= tol:
                    stopcond = True
            else:
                # stop if total number of iterations performed
                if self._iters >= self._maxiters:
                    stopcond = True


    def _init_centroids(self,data):
        
        centroids = np.empty((self.K,self.D),dtype=data.dtype)
        random_init = np.random.randint(0,self.N,self.K)
        self.init_seed = random_init
        
        for k in xrange(self.K):
            centroids[k] = data[random_init[k]]
        
        #self.centroids = centroids
        
        return centroids

    def _calc_dists(self,data,centroids):
        if self._mode == "cuda":
            dist_mat = self._cu_calc_dists(data,centroids,gridDim=None,
                                           blockDim=None)#,memManage='manual')
        elif self._mode == "numpy":
            dist_mat = self._np_calc_dists(data,centroids)

        elif self._mode == "python":
            dist_mat = self._py_calc_dists(data,centroids)
        
        
        return dist_mat
            
    def _py_calc_dists(self,data,centroids):
        N,D = data.shape
        K,cD = centroids.shape

        dist_mat = np.empty((N,K),dtype=data.dtype)

        for n in xrange(N):
            for k in xrange(K):
                dist=0
                for d in xrange(D):
                    diff = data[n,d]-centroids[k,d]
                    dist += diff ** 2
                dist_mat[n,k]=dist
                
        return dist_mat
            
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
                       keepDataRef=True):
        """
        TODO:
            - deal with gigantic data / distance matrix
            - deal with heavely assymetric distance matrix
                - if the number of blocks on any given dimension of 
                the grid > 2**16, divide that dimension by another dimension
                - don't forget to change the index computation in the kernel
            - deal with wide matrices
                - now it only works if matrix width is <=28
        """
        
      
        
        N,D = data.shape
        K,cD = centroids.shape
        
        #self.cuda_mem = memManage

        if self._cuda_mem not in ('manual','auto'):
            raise Exception("cuda_mem = \'manual\' or \'auto\'")
        
        
        
        if gridDim is not None and blockDim is not None:
            self._gridDim = gridDim
            self._blockDim = blockDim
            
        if self._gridDim is None or self._blockDim is None:
            #dists shape
            

            MAX_THREADS_BLOCK = 512 # GT520M has 48 CUDA cores
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

            # if grid dimension is bigger than MAX_GRID_XYZ_DIM,
            # the grid columns must be broken down in several along
            # the other grid dimensions
            if gridHeight > MAX_GRID_XYZ_DIM:
                self._dist_kernel = 1 # change kernel to use
                
                tpg = blockWidth * blockHeight # threads per block
                bpg = np.ceil( np.float(N * K) / tpg ) # blocks needed
                
                # number of grid columns
                gridWidth = np.ceil(bpg / MAX_GRID_XYZ_DIM)
                # number of grid rows
                gridHeight = np.ceil(bpg / gridWidth)

    
            blockDim = blockWidth, blockHeight
            gridDim = np.int(gridWidth), np.int(gridHeight)
        
            self._blockDim = blockDim
            self._gridDim = gridDim        
        
        distShape =  N,K
        dist_mat = np.empty(distShape,dtype=data.dtype)
        
        if self._cuda_mem == 'manual':
            
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
            
            self._cu_dist_kernel(dData,dCentroids,dDists,self._gridDim,self._blockDim)        
                
            dDists.copy_to_host(ary=dist_mat)
            numbapro.cuda.synchronize()

        elif self._cuda_mem == 'auto':
            self._cu_dist_kernel(data,centroids,dist_mat,self._gridDim,self._blockDim) 
        
        return dist_mat
        
    def _cu_dist_kernel(self,a,b,c,gridDim,blockDim):
        """
        Wraper to choose between kernels.
        """

        try:
            if self._dist_kernel == 0:
                self._cu_dist_kernel_normal[gridDim,blockDim](a,b,c)
            elif self._dist_kernel == 1:
                self._cu_dist_kernel_special_grid[gridDim,blockDim](a,b,c)
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            #print "*** print_tb:"
            #traceback.print_tb(exc_traceback, file=sys.stdout)
            print "*** print_exception:"
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      file=sys.stdout)

    
    @numbapro.cuda.jit("void(float32[:,:], float32[:,:], float32[:,:])")
    def _cu_dist_kernel_normal(a,b,c):
        k,n = numbapro.cuda.grid(2)

        ch, cw = c.shape # c width and height

        if n >= ch or k >= cw:
            return

        dist = 0.0
        for d in range(a.shape[1]):
            diff = a[n,d]-b[k,d]
            dist += diff ** 2
        c[n,k]= dist
        
    @numbapro.cuda.jit("void(float32[:,:], float32[:,:], float32[:,:])")
    def _cu_dist_kernel_special_grid(a,b,c):
        """
        This kernel can handle very 
        TODO:
        - fix for wide matrix
        """
        
        ## long vertical matrix thread index
        # thread ID inside block
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # block ID
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y

        # block dimensions
        bw = cuda.blockDim.x
        bh = cuda.blockDim.y

        # grid dimensions
        gw = cuda.gridDim.x
        gh = cuda.gridDim.y

        # compute thread's x and y index (i.e. datapoint and cluster)
        k = tx # block width is the same as matrix width
        # the second column of blocks means we want to add
        # 2**16 to the index
        n = ty + by * bh + bx*gh*bh

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
        
        if self._converge:
            all_dists = dist_mat.min(axis=1)
            inertia = all_dists.sum()

            self._error = self.inertia_ - inertia_
            self.inertia_ = inertia


        
        grouped_data=[[] for i in xrange(K)]
        
        for n in xrange(N):
            # add datum i to its assigned cluster assign[i]
            grouped_data[assign[n]].append(data[n])
        
        for k in xrange(K):
            grouped_data[k] = np.array(grouped_data[k])
        
        return assign,grouped_data
        
        
        def _assign_compute_centroids(self,data,dist_mat):
        
        N,K = dist_mat.shape
        
        assign = np.argmin(dist_mat,axis=1)
        
        if self._converge:
            all_dists = dist_mat.min(axis=1)
            inertia = all_dists.sum()

            self._error = self.inertia_ - inertia_
            self.inertia_ = inertia


        
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


    def _cu_mean(self):

        pass