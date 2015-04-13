# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:54:02 2015

@author: Diogo Silva
"""

import numpy as np
import numbapro
from numbapro import *  

import sys, traceback
from timeit import default_timer as timer


class K_Means:       
       
    
    def __init__(self,N=None,D=None,K=None):
        self.N = N
        self.D = D
        self.K = K
        

        self._mode = "cuda" #label mode
        self._centroid_mode = "index"
        
        self._error = None

        # cuda stuff
        self._cudaDataHanle = None
        self._cudaLabelsHanle = None
        self._cudaCentroidHandle = None
        
        self._cuda = True
        self._cuda_mem = "auto"

        self._dist_kernel = 0 # 0 = normal index, 1 = special grid index
        
        self._gridDim = None
        self._blockDim = None
        self._MAX_THREADS_BLOCK = 512
        self._MAX_GRID_XYZ_DIM = 65535

        # outputs
        self.inertia_ = None
        self._iters = 3
        self._converge = False       

        
    def fit_debug(self, data, K, iters=3, mode="cuda", cuda_mem='manual',tol=1e-4,max_iters=300):
        
        N,D = data.shape
            
        self.N = N
        self.D = D
        self.K = K
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
        
        print data.dtype, self.centroids.dtype
        
        while not stopcond:
            self.labels = self._cu_label(data,self.centroids)
            self.labels2 = self._np_label(data,self.centroids)

            print "cu_labels == np_labels:",np.allclose(self.labels,self.labels2)
            
            self.centroids =  self._np_recompute_centroids_group(data,self.centroids,self.labels2)
            self.centroids2 =  self._np_recompute_centroids_index(data,self.centroids,self.labels2)

            print "centroids_group == centroids_index:",np.allclose(self.centroids,self.centroids2)

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
            labels = self._label(data,self.centroids)
            
            #assign,grouped_data = self._assign_data(data,dist_mat)
            self.centroids =  self._recompute_centroids(data,self.centroids,labels)

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
        
        self.labels_ = labels

    def _recompute_centroids(self,data,centroids,labels):
        if self._centroid_mode == "group":
            new_centroids = self._np_recompute_centroids_group(data,centroids,labels)
        elif self._centroid_mode == "index":
            new_centroids = self._np_recompute_centroids_index(data,centroids,labels)
        elif self._centroid_mode == "iter":
            new_centroids = self._np_recompute_centroids_iter(data,centroids,labels)
        else:
            raise Exception("centroid mode invalid:",self._centroid_mode)

        return new_centroids

    def _init_centroids(self,data):
        
        centroids = np.empty((self.K,self.D),dtype=data.dtype)
        random_init = np.random.randint(0,self.N,self.K)
        self.init_seed = random_init
        
        for k in xrange(self.K):
            centroids[k] = data[random_init[k]]
        
        #self.centroids = centroids
        
        return centroids

    def _label(self,data,centroids):
        if self._mode == "cuda":
            labels = self._cu_label(data,centroids,gridDim=None,
                                           blockDim=None)#,memManage='manual')
        elif self._mode == "numpy":
            labels = self._np_label_fast(data,centroids)

        elif self._mode == "python":
            labels = self._py_label(data,centroids)
        
        
        return labels

    def _py_sqrd_euclidean(self,a,b):
        dist=0
        for d in xrange(a.shape[0]):
            dist += (a[d] - b[d])**2
        return dist
            
    def _py_label(self,data,centroids):
        """
        WARNING THIS IS COMPUTING          **** DISTANCE ****
        """

        N,D = data.shape
        K,cD = centroids.shape

        labels = np.zeros(N,dtype=np.int32)

        for n in xrange(N):

            # first iteration
            best_dist = self._py_sqrd_euclidean(data[n],centroids[0])
            best_label = 0

            for k in xrange(1,K):
                dist = self._py_sqrd_euclidean(data[n],centroids[k])
                if dist < best_dist:
                    best_dist = dist
                    best_label = k
            labels[n] = best_label
        return labels
            
    def _np_label(self,data,centroids):

        N,D = data.shape
        C,cD = centroids.shape

        labels = np.zeros(N,dtype=np.int32)
        
        # first iteration of all datapoints outside loop
        best_dist = data - centroids[0]
        best_dist = best_dist ** 2
        best_dist = best_dist.sum(axis=1)
        
        
        # remaining iterations
        for c in xrange(1,C):
            dist = data - centroids[c]
            dist = dist ** 2
            dist = dist.sum(axis=1)
            
            for n in xrange(N):
                if dist[n] < best_dist[n]:
                    best_dist[n] = dist[n]
                    labels[n] = c
        return labels
        
    def _np_label_fast(self,data,centroids):

        N,D = data.shape
        C,cD = centroids.shape

        labels = np.zeros(N,dtype=np.int32)
        
        # first iteration of all datapoints outside loop
        best_dist = data - centroids[0]
        best_dist = best_dist ** 2
        best_dist = best_dist.sum(axis=1)


        for c in xrange(1,C):
            dist = data - centroids[c]
            dist = dist ** 2
            dist = dist.sum(axis=1)
            
            thisCluster = np.full(N,c,dtype=np.int32)
            labels = np.where(dist < best_dist,thisCluster,labels)
            best_dist = np.where(dist < best_dist,dist,best_dist)
            
        return labels

    def _np_label_fast_2(self,data,centroids):
        """
        #TODO:
        - user nditer

        """
        N,D = data.shape
        C,cD = centroids.shape

        labels = np.zeros(N,dtype=np.int32)
        
        # first iteration of all datapoints outside loop
        best_dist = data - centroids[0]
        best_dist = best_dist ** 2
        best_dist = best_dist.sum(axis=1)


        for c in xrange(1,C):
            dist = data - centroids[c]
            dist = dist ** 2
            dist = dist.sum(axis=1)
            
            thisCluster = np.full(N,c,dtype=np.int32)
            labels = np.where(dist < best_dist,thisCluster,labels)
            best_dist = np.where(dist < best_dist,dist,best_dist)
            
        return labels

    
    def _cu_label(self,data,centroids,gridDim=None,blockDim=None,
                       keepDataRef=True):

        N,D = data.shape
        K,cD = centroids.shape
        
        if self._cuda_mem not in ('manual','auto'):
            raise Exception("cuda_mem = \'manual\' or \'auto\'")
        
        
        
        if gridDim is not None and blockDim is not None:
            self._gridDim = gridDim
            self._blockDim = blockDim
            
        if self._gridDim is None or self._blockDim is None:
            #dists shape
            
            blockHeight = self._MAX_THREADS_BLOCK
            blockDim = 1, blockHeight
            
            tpb = np.prod(blockDim)
            bpg = np.int(np.ceil(np.float(N) / tpb))
            

            # if grid dimension is bigger than MAX_GRID_XYZ_DIM,
            # the grid columns must be broken down in several along
            # the other grid dimensions
            if bpg > self._MAX_GRID_XYZ_DIM:
                # number of grid columns
                gridWidth = np.ceil(bpg / self._MAX_GRID_XYZ_DIM)
                # number of grid rows
                gridHeight = np.ceil(bpg / gridWidth)    
            
                gridDim = np.int(gridWidth), np.int(gridHeight)
            else:
                gridDim = 1,bpg
                
                
            self._blockDim = blockDim
            self._gridDim = gridDim        
        

        labels = np.empty(N,dtype=np.int32)
        
        
        if self._cuda_mem == 'manual':
            # copy data and centroids, allocate memory
            if keepDataRef:
                if self._cudaDataHanle is None:
                    dData = cuda.to_device(data)
                    self._cudaDataHanle = dData
                else:
                    dData = self._cudaDataHanle
            else:
                dData = cuda.to_device(data)
                
            dCentroids = cuda.to_device(centroids)
            dLabels = numbapro.cuda.device_array_like(labels)
            
            self._cudaLabelsHandle = dLabels
            self._cudaCentroidsHandle = dCentroids
            
            self._cu_label_kernel(dData,dCentroids,dLabels,self._gridDim,self._blockDim)        
            
            numbapro.cuda.synchronize()
            dLabels.copy_to_host(ary=labels)
            

        elif self._cuda_mem == 'auto':
            self._cu_label_kernel(data,centroids,labels,self._gridDim,self._blockDim) 
        
        return labels
        
    def _cu_label_kernel(self,a,b,c,gridDim,blockDim):
        """
        Wraper to choose between kernels.
        """
        
        if self._iters == "converge":
            self._cu_label_kernel_dists[gridDim,blockDim](a,b,c,self.dists)
        else:
            self._cu_label_kernel_normal[gridDim,blockDim](a,b,c)


        """try:
            
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            #print "*** print_tb:"
            #traceback.print_tb(exc_traceback, file=sys.stdout)
            print "*** print_exception:"
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      file=sys.stdout)"""


    # data, centroids, labels
    @numbapro.cuda.jit("void(float32[:,:], float32[:,:], int32[:])")
    def _cu_label_kernel_normal(a,b,c):
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
        # tx doesn't matter
        # the second column of blocks means we want to add
        # 2**16 to the index
        n = ty + by * bh + bx*gh*bh

        N = c.shape[0] # number of datapoints
        K,D = b.shape # centroid shape

        if n >= N:
            return

        # first iteration outside loop
        dist = 0.0
        for d in range(D):
            diff = a[n,d]-b[0,d]
            dist += diff ** 2

        best_dist = dist
        best_label = 0

        # remaining iterations
        for k in range(1,K):

            dist = 0.0
            for d in range(D):
                diff = a[n,d]-b[k,d]
                dist += diff ** 2

            if dist < best_dist:
                best_dist = dist
                best_label = k

        c[n] = best_label


    # data, centroids, labels
    @numbapro.cuda.jit("void(float32[:,:], float32[:,:], int32[:], float32[:])")
    def _cu_label_kernel_dists(a,b,c,dists):
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
        # tx doesn't matter
        # the second column of blocks means we want to add
        # 2**16 to the index
        n = ty + by * bh + bx*gh*bh

        N = c.shape[0] # number of datapoints
        K,D = b.shape # centroid shape

        if n >= N:
            return

        # first iteration outside loop
        dist = 0.0
        for d in range(D):
            diff = a[n,d]-b[0,d]
            dist += diff ** 2

        best_dist = dist
        best_label = 0

        # remaining iterations
        for k in range(1,K):

            dist = 0.0
            for d in range(D):
                diff = a[n,d]-b[k,d]
                dist += diff ** 2


            if dist < best_dist:
                best_dist = dist
                best_label = k

        c[n] = best_label
        dists[n] = best_dist

    def _cu_centroids(data,centroids,labels):
        pass

    
    # data, centroids, labels, centroid counter, centroid sum
    @numbapro.cuda.jit("void(float32[:,:], float32[:,:], int32[:], int32[:], float32[:])")
    def _cu_centroids_kernel_normal(data,centroids,labels,cCounter,cSum):
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




    def _np_recompute_centroids_group(self,data,centroids,labels):
        """
        Iterates over data. Makes a list of data for each cluster.
        Transforms 
        """

        N,D = data.shape
        K,D = centroids.shape
        
        grouped_data=[[] for i in xrange(K)] 

        new_centroids = np.zeros_like(centroids)
        
        for n in xrange(N):
            # add datum i to its assigned cluster assign[i]
            grouped_data[labels[n]].append(data[n])
            
        for k in xrange(K):
            #grouped_data[k] = np.array(grouped_data[k])
            new_centroids[k] = np.mean(grouped_data[k],axis=0)

        return new_centroids

    def _np_recompute_centroids_index(self,data,centroids,labels):
        """
        This only works when every cluster has some point.
        This method sorts the data so that data points belonging to 
        the same cluster are next to each other. The labels are organized
        in the same order. Then we get the indeces where the labels change.
        Those indeces are then used to partition the data and compute the mean.

        It's very fast.
        A major drawback is that it crashes on empty clusters. It also
        replicates data.
        """
        # change to get dimension from class or search a non-empty cluster
        #dim = grouped_data[0][0].shape[1]
        N,D = data.shape
        K,D = centroids.shape       
        
        new_centroids = centroids.copy()
        centroid_count = np.zeros(K,dtype=np.int32)

        # sort labels and data by cluster
        labels_sorted = labels.argsort()
        labels = labels[labels_sorted]
        sortedData = data[labels_sorted]

        # changed indeces array
        labelChangedIndex = np.where(labels[1:] != labels[:-1])[0] + 1

        # print "\nlabelChangedIndex\t",labelChangedIndex
  
        #
        #  DEALS WITH EMPTY CLUSTERS
        #

        nonEmptyClusters = labelChangedIndex.shape[0]

        # first iteration
        startIndex,endIndex = 0,labelChangedIndex[0]
        clusterID = labels[startIndex]
        new_centroids[clusterID] = sortedData[startIndex:endIndex].mean(axis=0)

        # middle iterations
        for k in xrange(1,nonEmptyClusters-1):
            startIndex,endIndex = labelChangedIndex[k-1],labelChangedIndex[k]
            clusterID = labels[startIndex]
            new_centroids[clusterID] = sortedData[startIndex:endIndex].mean(axis=0)

        # last iteration
        startIndex = labelChangedIndex[-1]
        clusterID = labels[startIndex]
        new_centroids[clusterID] = sortedData[startIndex:].mean(axis=0)

        """
        #
        #  DOESN'T DEAL WITH EMPTY CLUSTERS
        #

        # first iteration
        startIndex,endIndex = 0,labelChangedIndex[0]
        new_centroids[0] = sortedData[startIndex:endIndex].mean(axis=0)

        # middle iterations
        for k in xrange(1,K-1):
            startIndex,endIndex = labelChangedIndex[k-1],labelChangedIndex[k]
            new_centroids[k] = sortedData[startIndex:endIndex].mean(axis=0)

        # last iteration
        startIndex = labelChangedIndex[-1]
        new_centroids[-1] = sortedData[startIndex:].mean(axis=0)
        """


        return new_centroids
    
    def _np_recompute_centroids_iter(self,data,centroids,labels):
        """
        INEFFICIENT
        """
        # change to get dimension from class or search a non-empty cluster
        #dim = grouped_data[0][0].shape[1]
        N,D = data.shape
        K,D = centroids.shape       
        
        new_centroids = np.zeros_like(centroids)
        centroid_count = np.zeros(K,dtype=np.int32)
        
        """
        for n, l in np.nditer([data,labels]):
            new_centroids[l] += n
            centroid_count[l] += 1

        """
        """
        it = np.nditer(labels, flags=['f_index'])
        while not it.finished:
            new_centroids[it.value] += data[it.index]
            centroid_count[it.value] += 1
            it.iternext()
        """

        # sum of all data points in each cluster
        for n in xrange(N):
            new_centroids[labels[n]] += data[n]
            centroid_count[labels[n]] += 1       

        
        # compute mean
        new_centroids = new_centroids / centroid_count.reshape((K,1))
            
        return new_centroids
