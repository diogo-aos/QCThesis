# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:54:02 2015

@author: Diogo Silva

TODO:
- test cuda return distances
- implement cuda distance reduce job
- converge mode in all label functions (low priority since those are not in use)
- improve cuda labels with local block memory
- make sure that cuda labels returns the distance array every iteration 
  (centroid computation needs it)
"""

import numpy as np
from numba import jit, cuda, int32, float32, void

from random import sample

from MyML.utils.sorting import arg_k_select

#import sys, traceback
#from timeit import default_timer as timer


class K_Means:       
       
    def __init__(self, n_clusters=8, mode="cuda", cuda_mem='auto', tol=1e-4,
                 max_iter=300, init='random'):
        self.n_clusters = n_clusters

        self._mode = mode #label mode
        self._centroid_mode = "good" #recompute centroids mode

        # TODO check parameters, check iters iterss numberf or "converge"


        # check if centroids are supplied
        if init == 'random':
            self._centroid_type = init
        elif type(init) is np.ndarray:
            if init.shape[0] != n_clusters:
                raise Exception("Number of clusters indicated different \
                                 from number of centroids supplied.")
            self._centroid_type = "supplied"
            self.centroids = init
        else:
            raise ValueError('Centroid may be \'random\' or an ndarray \
                              containing the centroids to use.')

        # execution flow
        self.tol = tol
        self.max_iter = max_iter

        self._converge = True
        self._last_iter = False
        

        # outputs
        self.inertia_ = np.inf
        self.iters_ = 0

        # cuda stuff
        self._cudaDataHandle = None
        self._cudaLabelsHandle = None
        self._cudaCentroidHandle = None
        
        self._cuda = True
        self._cuda_mem = cuda_mem

        self._dist_kernel = 0 # 0 = normal index, 1 = special grid index
        
        self._gridDim = None
        self._blockDim = None
        self._MAX_THREADS_BLOCK = 512
        self._MAX_GRID_XYZ_DIM = 65535
        self._CUDA_WARP = 32

        self._PPT = 1 # points to process per thread

    def fit(self, data):

        if data.dtype != np.float32:
            print "WARNING DATA DUPLICATION: data converted to float32. \
                   TODO: accept other formats"
            data = data.astype(np.float32)
        
        N,D = data.shape
            
        self.N = N
        self.D = D        
        
        # if random centroids, than get them
        # otherwise they're already there
        if self._centroid_type == "random":
            self.centroids = self._init_centroids(data)

        # reset variables for flow control
        stopcond = False
        self.iters_ = 0
        self.inertia_ = np.inf
        self._last_iter = False # this is only for labels centroid recomputation


        self._dists = np.empty(self.N, dtype = np.float32)

        while not stopcond:
            # compute labels
            labels = self._label(data,self.centroids)

            self.iters_ += 1 #increment iteration counter

            ## evaluate stop conditions
            # convergence condition
            if self._converge:
                # compute new inertia
                new_inertia = self._dists.sum()

                # compute error
                error = np.abs(new_inertia - self.inertia_)
                self._error = error
                # save new inertia
                self.inertia_ = new_inertia

                # stop if convergence tolerance achieved
                if error <= self.tol:
                    stopcond = True
                    self._last_iter = True

            # iteration condition
            if self.iters_ >= self.max_iter:
                stopcond = True
                self._last_iter = True

            if stopcond:
                break
            # compute new centroids
            self.centroids = self._recompute_centroids(data,self.centroids,labels)
        
        self.labels_ = labels
        self.cluster_centers_ = self.centroids

    def _init_centroids(self, data):
        
        #centroids = np.empty((self.n_clusters,self.D),dtype=data.dtype)
        #random_init = np.random.randint(0,self.N,self.n_clusters)
        
        random_init = sample(xrange(self.N), self.n_clusters)
        #self.init_seed = random_init

        centroids = data[random_init]

        return centroids

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #               COMPUTE LABELS                                # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _label(self,data,centroids):
        """
        results is a tuple of labels (pos 0) and distances (pos 1) when
        self._converge == True
        """

        # we need array for distances to check convergence
        if self._mode == "cuda":
            labels = self._cu_label(data, centroids)
        elif self._mode == "special": #for tests only
            labels=np.empty(self.N, dtype=np.int32)
            self._cu_label_kernel(data,centroids,labels,[1,512],[1,59])
        elif self._mode == "numpy":
            labels = self._np_label_fast(data,centroids)
        elif self._mode == "numba":
            labels, dists = numba_label(data,centroids)
            self._dists = dists
        elif self._mode == "python":
            labels = self._py_label(data,centroids)

        return labels

    def _py_sqrd_euclidean(self,a,b):
        """
        Euclidean distance between points a and b.
        """
        dist=0
        for d in xrange(a.shape[0]):
            dist += (a[d] - b[d])**2
        return dist

    def _py_label(self,data,centroids):

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

                    if self._converge:
                        self._dists[n] = best_dist

            labels[n] = best_label

        return labels

    @jit(float32(float32[:],float32[:]))
    def _numba_euclid(a,b):
        dist = 0
        for d in range(a.shape[0]):
            dist += (a[d] - b[d]) ** 2
        return dist
            
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
        """
        uses more memory
        """
        N,D = data.shape
        C,cD = centroids.shape


        labels = np.zeros(N,dtype=np.int32)

        # first iteration of all datapoints outside loop
        # distance from points to centroid 0
        best_dist = data - centroids[0]
        best_dist = best_dist ** 2
        best_dist = best_dist.sum(axis=1) 


        for c in xrange(1,C):
            # distance from points to centroid c
            dist = data - centroids[c]
            dist = dist ** 2
            dist = dist.sum(axis=1)
            
            #thisCluster = np.full(N,c,dtype=np.int32)
            #labels = np.where(dist < bestd_ist,thisCluster,labels)
            labels[dist<best_dist] = c
            best_dist = np.minimum(dist,best_dist)


        #if self._converge:
        self._dists = best_dist

        return labels

    def _np_label_fast_2(self,data,centroids):
        """
        slower than fast

        #TODO:
        - use nditer

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

    def _compute_cuda_dims(self, data, use2d = False):

        N, D = data.shape

        if use2d:
            blockHeight = self._MAX_THREADS_BLOCK
            blockWidth = 1
            blockDim = blockWidth, blockHeight

            # threads per block
            tpb = np.prod(blockDim)

            # blocks per grid = data cardinality divided by number
            # of threads per block (1 thread - 1 data point)
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
        else:
            blockDim = self._MAX_THREADS_BLOCK
            points_in_block = self._MAX_THREADS_BLOCK * self._PPT
            bpg = np.float(N) / points_in_block
            gridDim = np.int(np.ceil(bpg))
            
            
        self._blockDim = blockDim
        self._gridDim = gridDim
    
    def _cu_label(self,data,centroids):

        N,D = data.shape
        K,cD = centroids.shape
        
        if self._cuda_mem not in ('manual','auto'):
            raise Exception("cuda_mem = \'manual\' or \'auto\'")
            
        if self._gridDim is None or self._blockDim is None:
            self._compute_cuda_dims(data)       
        
        labels = np.empty(N, dtype = np.int32)
        
        if self._cuda_mem == 'manual':
            # copy dataset and centroids, allocate memory

            # avoids redundant data transfer
            # if dataset has not been sent to device, send it and save handle
            if self._cudaDataHandle is None:
                dData = cuda.to_device(data)
                self._cudaDataHandle = dData
            # otherwise just use handle
            else:
                dData = self._cudaDataHandle

            # copy centroids to device
            dCentroids = cuda.to_device(centroids)

            # allocate array for labels and dists
            dLabels = cuda.device_array_like(labels)
            dDists = cuda.device_array_like(self._dists)
            
            #self._cudaLabelsHandle = dLabels
            #self._cudaCentroidsHandle = dCentroids
            
            _cu_label_kernel_dists[self._gridDim,self._blockDim](dData, 
                                                                 dCentroids, 
                                                                 dLabels, 
                                                                 dDists)
            
            # synchronize threads before copying data
            #cuda.synchronize()

            # copy labels from device to host
            dLabels.copy_to_host(ary = labels)
            # copy distance to centroids from device to host
            dists = dDists.copy_to_host()
            self._dists = dists

        elif self._cuda_mem == 'auto':
            _cu_label_kernel_dists[self._gridDim,self._blockDim](data, 
                                                                centroids, 
                                                                labels, 
                                                                self._dists)

        else:
            raise ValueError("CUDA memory management type may either \
                              be \'manual\' or \'auto\'.")
        
        return labels
        
    def _cu_label_kernel(self,a,b,c,d,gridDim,blockDim):
        """
        Wraper to choose between kernels.
        """
        # if converging and manual memory management, use distance handle
        if self._cuda_mem == 'manual':
            self._cu_label_kernel_dists[gridDim,blockDim](a,b,c,d)
        # if converging and auto memory management, use distance array
        else:
            self._cu_label_kernel_dists[gridDim,blockDim](a,b,c,d)
        pass

        # """try:
            
        # except Exception:
        #     exc_type, exc_value, exc_traceback = sys.exc_info()
        #     #print "*** print_tb:"
        #     #traceback.print_tb(exc_traceback, file=sys.stdout)
        #     print "*** print_exception:"
        #     traceback.print_exception(exc_type, exc_value, exc_traceback,
        #                               file=sys.stdout)"""


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                      RECOMPUTE CENTROIDS                    # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _recompute_centroids(self,data,centroids,labels):
        if self._centroid_mode == "group":
            new_centroids = self._np_recompute_centroids_group(data,centroids,labels)
        elif self._centroid_mode == "index":
            new_centroids = self._np_recompute_centroids_index(data,centroids,labels)
        elif self._centroid_mode == "iter":
            new_centroids = self._np_recompute_centroids_iter(data,centroids,labels)
        elif self._centroid_mode == "good":
            new_centroids = self._np_recompute_centroids_good(data,centroids,labels) 
        elif self._centroid_mode == "good_numba":
            new_centroids = numba_recompute_centroids_good(data, centroids, labels, self._dists)
        else:
            raise Exception("centroid mode invalid:",self._centroid_mode)

        return new_centroids

    def _cu_centroids(data,centroids,labels):
        pass


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

    def _np_recompute_centroids_good(self,data,centroids,labels):
        """
        this version doesn't discard clusters; instead it uses the same scheme
        as sci-kit learn
        """
        # change to get dimension from class or search a non-empty cluster
        #dim = grouped_data[0][0].shape[1]
        N,D = data.shape
        K,D = centroids.shape       
        
        #new_centroids = centroids.copy()
        new_centroids = np.zeros_like(centroids)

        nonEmptyClusters = np.unique(labels)

        n_emptyclusters = K - nonEmptyClusters.size
        furtherDistsArgs = self._dists.argsort()[::-1][:n_emptyclusters]

        j=0 #empty cluster indexer
        for i in xrange(K):
            if i in nonEmptyClusters:
                new_centroids[i] = data[labels==i].mean(axis=0)
            else:
                new_centroids[i] = data[furtherDistsArgs[j]]
                j+=1

        return new_centroids

    @jit(void(float32[:],float32[:],int32[:]))
    def _numba_recompute_centroids(data, centroids, labels):
        N,D = data.shape
        K,D = centroids.shape

        count = np.zeros(K, dtype = np.int32)

        for k in range(K):
            for d in range(D):
                centroids[k,d] = 0

        for n in range(N):
            k = labels[n] # centroid to use
            count[k] += 1
            for d in range(D):

                pass

    def _np_recompute_centroids_index(self,data,centroids,labels):
        """
        This only works when every cluster has some point.
        This method sorts the data so that data points belonging to 
        the same cluster are next to each other. The labels are organized
        in the same order. Then we get the indeces where the labels change.
        Those indeces are then used to partition the data and compute the mean.

        It's very fast.
        It replicates data. Bad if data is huge. Workarounds:
        - sort in place.
        - don't sort data, get mean from indexing original data directly
        TODO: check which option is faster
        """
        # change to get dimension from class or search a non-empty cluster
        #dim = grouped_data[0][0].shape[1]
        N,D = data.shape
        K,D = centroids.shape       
        
        #new_centroids = centroids.copy()
        new_centroids = np.zeros_like(centroids)

        # sort labels and data by cluster
        labels_sorted = labels.argsort()
        labels = labels[labels_sorted]
        sortedData = data[labels_sorted]

        # array storing the dataset indices where the clustering changes 
        # (after the it has been ordered) this stores every i-th index where 
        # the i-th label is different from the (i+1)-th label
        labelChangedIndex = np.where(labels[1:] != labels[:-1])[0] + 1

        #
        #  DEALS WITH EMPTY CLUSTERS
        #

        # number of empty clusters is equal to the number of indices that 
        # partition the labels plus 1
        nonEmptyClusters = labelChangedIndex.shape[0] + 1

        if nonEmptyClusters == 1:
            return data.mean(axis=0).reshape(1,data.shape[1])

        ## first iteration
        # start and end index of first cluster in labels
        try:
            startIndex,endIndex = 0,labelChangedIndex[0]
        except IndexError:
            print "labels: ",labels
            print "centroids: ",centroids
            print "centroids equal:",(centroids[0]==centroids[1])
            print "labelChangedIndex: ",labelChangedIndex
            print labelChangedIndex[0] #generate original error

        # the cluster number is given by the label of the start index
        clusterID = labels[startIndex]

        # slice the data and compute the mean
        new_centroids[clusterID] = sortedData[startIndex:endIndex].mean(axis=0)

        # add cluster to final partition
        # should only be executed in last iteration of K-Means
        if self._last_iter:
            self.partition=list()
            self.partition.append(labels_sorted[startIndex:endIndex])

        ## middle iterations
        for k in xrange(1,nonEmptyClusters-1):
            startIndex,endIndex = labelChangedIndex[k-1],labelChangedIndex[k]
            clusterID = labels[startIndex]
            new_centroids[clusterID] = sortedData[startIndex:endIndex].mean(axis=0)

             # add clusters to final partition
            if self._last_iter:
                self.partition.append(labels_sorted[startIndex:endIndex])

        # last iteration
        startIndex = labelChangedIndex[-1]
        clusterID = labels[startIndex]
        new_centroids[clusterID] = sortedData[startIndex:].mean(axis=0)

        # add cluster to final partition
        if self._last_iter:
            self.partition.append(labels_sorted[startIndex:])

        # remove empty clusters
        emptyClusters = [i for i,c in enumerate(centroids) if not c.any()]
        new_centroids = np.delete(new_centroids,emptyClusters,axis=0)

        if np.unique(new_centroids).shape[0] == 0:
            print "centroids: ",new_centroids
            raise ValueError("centroids empty.")

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


# data, centroids, labels
@cuda.jit("void(float32[:,:], float32[:,:], int32[:])")
def _cu_label_kernel_normal(a,b,c):

    """
    Computes the labels of each data point without storing the distances.
    """
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


CUDA_PPT = 1
# data, centroids, labels
@cuda.jit("void(float32[:,:], float32[:,:], int32[:], float32[:])")
def _cu_label_kernel_dists(a,b,c,dists):

    """
    Computes the labels of each data point storing the distances.
    """

    # # thread ID inside block
    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y

    # # block ID
    # bx = cuda.blockIdx.x
    # by = cuda.blockIdx.y

    # # block dimensions
    # bw = cuda.blockDim.x
    # bh = cuda.blockDim.y

    # # grid dimensions
    # gw = cuda.gridDim.x
    # gh = cuda.gridDim.y

    # compute thread's x and y index (i.e. datapoint and cluster)
    # tx doesn't matter
    # the second column of blocks means we want to add
    # 2**16 to the index
    #n = ty + by * bh + bx*gh*bh

    tgid = cuda.grid(1) * CUDA_PPT

    N = c.shape[0] # number of datapoints
    K,D = b.shape # centroid shape

    if tgid >= N:
        return

    for n in range(tgid, tgid + CUDA_PPT):

        if n >= N:
            return

        # first iteration outside loop
        dist = 0.0
        for d in range(D):
            diff = a[n,d] - b[0,d]
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

@jit(nopython=True)
def numba_label(data, centroids):

    N = data.shape[0]
    K,D = centroids.shape

    labels = np.empty(N, dtype=np.int32)
    dists = np.empty(N, dtype=np.float32)

    for n in range(0, N):

        # first iteration outside loop
        dist = 0.0
        for d in range(D):
            diff = data[n,d] - centroids[0,d]
            dist += diff ** 2

        best_dist = dist
        best_label = 0

        # remaining iterations
        for k in range(1,K):

            dist = 0.0
            for d in range(D):
                diff = data[n,d]-centroids[k,d]
                dist += diff ** 2

            if dist < best_dist:
                best_dist = dist
                best_label = k

        labels[n] = best_label
        dists[n] = best_dist

    return labels, dists

@jit(nopython = True)
def numba_recompute_centroids_good(data, centroids, labels, dists):
    N = labels.size
    K, D = centroids.shape

    new_centroids = np.zeros((K,D), dtype=np.float32)       

    # count samples in clusters
    labels_bincount = np.zeros(K, dtype=np.int32)
    for n in range(N):
        l = labels[n]
        labels_bincount[l] += 1

    # check for empty clusters
    n_emptyClusters = 0
    for l in range(K):
        if labels_bincount[l] == 0:
            n_emptyClusters += 1

    # get farthest points from clusters (K-select)
    furtherDistsArgs = np.empty(n_emptyClusters, dtype=np.int32) 
    arg_k_select(dists, n_emptyClusters, furtherDistsArgs)
    # furtherDistsArgs = arg_k_select(dists, n_emptyClusters)
    
    # increment datapoints to respective centroids
    for n in range(N):
        n_label = labels[n]
        for d in range(D):
            new_centroids[n_label,d] += data[n,d]

    i = 0
    for k in range(K):
        if labels_bincount[k] != 0: # compute final centroid
            for d in range(D):
                new_centroids[k, d] /= labels_bincount[k]
        else: # centroid will be one of furthest points
            i_arg = furtherDistsArgs[i]
            for d in range(D):
                new_centroids[k, d] = data[i_arg, d]

    return new_centroids

# data, centroids, labels, centroid counter, centroid sum
# @cuda.jit("void(float32[:,:], float32[:,:], int32[:], int32[:], float32[:])")
@cuda.jit(void(float32[:,:], float32[:,:], int32[:], int32[:], float32[:]))
# @cuda.jit
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

    pass