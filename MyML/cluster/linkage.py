# -*- coding: utf-8 -*-
"""
Created on 15-06-2015

@author: Diogo Silva

"""
import numpy as np
from numba import cuda, jit, void, int32, float32
from MyML.graph.mst import boruvka_minho_seq, boruvka_minho_gpu,\
                           compute_cuda_grid_dim
from MyML.graph.connected_components import connected_comps_seq,\
                                            connected_comps_gpu
from MyML.graph.build import getGraphFromEdges_gpu, getGraphFromEdges_seq

from numbapro.cudalib.sorting import RadixSort
from numbapro.cudalib.cublas import Blas

import scipy_numba.cluster._hierarchy_eac as hie_eac


def sl_mst_lifetime_seq(dest, weight, fe, od, disconnect_weight = None):

    if disconnect_weight is None:
        disconnect_weight = weight.max()

    mst, n_edges = boruvka_minho_seq(dest, weight, fe, od)

    # Get array with only the considered weights in the MST
    # and remove those edges in the MST edge list
    mst_weights = weight[mst[:n_edges]]

    # Sort the MST weights. There are no repeated edges at this
    # point since the output MST is like a directed graph.
    sortedWeightArgs = mst_weights.argsort()
    mst_weights = mst_weights[sortedWeightArgs]
    mst = mst[sortedWeightArgs]

    # Allocate array for the lifetimes.
    lifetimes = mst_weights[1:] - mst_weights[:-1]

    arg_max_lt = lifetimes.argmax()
    max_lt = lifetimes[arg_max_lt]

    # this is the lifetime between edges with no connection and the weakest link
    #lt_threshold = disconnect_weight - max_lt
    lt_threshold = disconnect_weight - mst_weights[-1]

    # if the maximum lifetime if higher or equal than the lifetime threshold
    # cut the tree
    if max_lt >= lt_threshold:
        # from arg_max_lt onward all edges are discarded
        n_discarded = lifetimes.size - arg_max_lt + 1
        
        # remove edges
        mst = mst[:-n_discarded]

    del lifetimes, mst_weights

    ndest = np.empty(mst.size * 2, dtype = dest.dtype)
    nweight = np.empty(mst.size * 2, dtype = weight.dtype)
    nfe = np.empty_like(fe)
    nod = np.zeros_like(od)

    # build graph from mst
    getGraphFromEdges_seq(dest, weight, fe, od, mst,
                          nod, nfe, ndest, nweight)

    labels = connected_comps_seq(ndest, nweight, nfe, nod)

    del ndest, nweight, nfe, nod

    return labels



def sl_mst_lifetime_gpu(dest, weight, fe, od, disconnect_weight = None,
                        MAX_TPB = 256, stream = None):
    """
    Input are device arrays.
    Inputs:
     dest, weight, fe 		: device arrays
     disconnect_weight 		: weight between unconnected vertices
     mst 					: list of edges in MST
     MAX_TPB 				: number of threads per block
     stream 				: CUDA stream to use
    TODO:
     - argmax is from cuBlas and only works with 32/64 floats. Make this work 
       with any type.
     - 
    """

    if disconnect_weight is None:
        disconnect_weight = weight.max()

    if stream is None:
        myStream = cuda.stream()
    else:
        myStream = stream

    mst, n_edges = boruvka_minho_gpu(dest, weight, fe, od,
                                     MAX_TPB=MAX_TPB, stream=myStream,
    	  							 returnDevAry=True)

    # Allocate array for the mst weights.
    h_n_edges = int(n_edges.getitem(0, stream=myStream)) # edges to keep in MST
    mst_weights = cuda.device_array(h_n_edges, dtype=weight.dtype)    

    # Get array with only the considered weights in the MST
    # and remove those edges in the MST edge list
    mstGrid = compute_cuda_grid_dim(h_n_edges, MAX_TPB)
    d_weight = cuda.to_device(weight, stream = myStream)
    getWeightsOfEdges_gpu[mstGrid, MAX_TPB, myStream](mst, n_edges, d_weight,
                                                      mst_weights)    

    # Sort the MST weights. There are no repeated edges at this
    # point since the output MST is like a directed graph.
    sorter = RadixSort(maxcount = mst_weights.size, dtype = mst_weights.dtype,
                       stream = myStream)
    sortedWeightArgs = sorter.argsort(mst_weights)

    # Allocate array for the lifetimes.
    lifetimes = cuda.device_array(mst_weights.size - 1, dtype=mst_weights.dtype)
    compute_lifetimes_CUDA[mstGrid, MAX_TPB, myStream](mst_weights, lifetimes)

    maxer = Blas(stream)
    arg_max_lt = maxer.amax(lifetimes)
    max_lt = lifetimes.getitem(arg_max_lt)

    # this is the lifetime between edges with no connection and the weakest link
    #lt_threshold = disconnect_weight - max_lt
    lt_threshold = disconnect_weight - mst_weights.getitem(mst_weights.size - 1)

    # if the maximum lifetime is higher or equal than the lifetime threshold
    # cut the tree
    if max_lt >= lt_threshold:
        # from arg_max_lt onward all edges are discarded
        n_discarded = lifetimes.size - arg_max_lt + 1

        # remove edges
        removeGrid = compute_cuda_grid_dim(n_discarded, MAX_TPB)
        removeEdges[removeGrid, MAX_TPB](edgeList, sortedArgs, n_discarded)

        # compute new amount of edges and update it
        new_n_edges = h_n_edges - n_discarded
        cuda.to_device(np.array([new_n_edges], dtype = n_edges.dtype),
                       to = n_edges,
                       stream = myStream)

    ngraph = getGraphFromEdges_gpu(dest, weight, fe, od, edges = mst,
                                   n_edges = n_edges, MAX_TPB = MAX_TPB,
                                   stream = myStream)

    ndest, nweight, nfe, nod = ngraph

    labels = connected_comps_gpu(ndest, nweight, nfe, nod,
                                 MAX_TPB = 512, stream = myStream)

    del ndest, nweight, nfe, nod, lifetimes

    return labels




@cuda.jit
def removeEdges(edgeList, sortedArgs, n_discarded):
    """
    inputs:
        edgeList         : list of edges
        sortedArgs         : argument list of the sorted weight list
        n_discarded     : number of edges to be discarded specified in sortedArgs

    Remove discarded edges form the edge list.
    Each edge discarded is replaced by -1.

    Discard edges specified by the last n_discarded arguments
    in the sortedArgs list.

    """

    tgid = cuda.grid(1)

    # one thread per edge that must be discarded
    # total number of edges to be discarded is the difference 
    # between the between the total number of edges and the 
    # number of edges to be considered + the number edges 
    # to be discarded

    if tgid >= n_discarded:
        return

    # remove not considered edges
    elif tgid < n_considered_edges:
        maxIdx = edgeList.size - 1 # maximum index of sortedArgs
        index = maxIdx - tgid # index of 
        edgeList[index] = -1




@cuda.jit
def argmax_lvl0(ary, reduce_max, reduce_arg):
    """
    This only works for positive values arrays.
    Shared memory must be initialized with double the size of 
    the block size.
    """
    sm_ary = cuda.shared.array(shape = 0, dtype = ary.dtype)

    # each thread will process two elements
    tgid = cuda.grid(1)
    thid = cuda.threadIdx.x

    # pointer to value and argument side of shared memory
    val_pointer = 0
    arg_pointer = sm_ary.size / 2    

    # when global thread id is bigger or equal than the ary size
    # it means that the block is incomplete; in this case we just
    # fill the rest of the block with -1 so it is smaller than all
    # other elements; this only works for positive arrays
    if tgid < ary.size:
        sm_ary[val_pointer + thid] = ary[tgid]
        sm_ary[arg_pointer + thid] = tgid
    else:
        sm_ary[val_pointer + thid] = 0
        sm_ary[arg_pointer + thid] = -1        


    cuda.syncthreads()

    s = cuda.blockDim.x / 2
    while s >0:
        index = 2 * s * thid

        if thid < s:
            # only change if the left element is smaller than the right one
            if sm_ary[val_pointer + thid] < sm_ary[val_pointer + thid + s]:
                sm_ary[val_pointer + thid] = sm_ary[val_pointer + thid + s]
                sm_ary[arg_pointer + index] = sm_ary[arg_pointer + index + s]

        cuda.syncthreads()

    if thid == 0:
        reduce_ary[cuda.blockIdx.x] = sm_ary[val_pointer]
        reduce_arg[cuda.blockIdx.x] = sm_ary[arg_pointer]

@cuda.jit
def argmax_lvl1(reduce_max, reduce_arg):
    pass

@cuda.jit
def search_argmin_val(ary, val):
    tgid = cuda.grid(1)
    if tgid >= ary.size:
        return

@cuda.reduce
def max_gpu(a,b):
    if a >= b:
        return a
    else:
        return b

@cuda.jit
def compute_lifetimes_CUDA(nweight, lifetimes):
    edge = cuda.grid(1)
    
    if edge >= lifetimes.size:
        return
    
    lifetimes[edge] = nweight[edge + 1] - nweight[edge]

@cuda.jit#("void(int32[:],int32[:],int32[:],int32[:])")
           # "void(int32[:],int32[:],float32[:],float32[:])"])
def getWeightsOfEdges_gpu(edges, n_edges, weights, nweights):
    """
    This function will take a list of edges (edges), the number of edges to 
    consider (n_edges, the weights of all the possible edges (weights) and the 
    array for the weights of the list of edges and put the weight of each edge 
    in the list of edges in the nweights, in the same position.

    The kernel will also discard not considered edges, i.e. edges whose 
    argument >= n_edges.
    Discarding an edge is done by replacing the edge by -1.
    """
    # n_edges_sm = cuda.shared.array(1, dtype = int32)
    edge = cuda.grid(1)

    if edge >= edges.size:
        return
    
    # if edge == 0:
    #     n_edges_sm[0] = n_edges[0]
    # cuda.syncthreads()
    
    
    # if edge >= n_edges_sm[0]:
    if edge >= n_edges[0]:
        edges[edge] = -1
    else:
        myEdgeID = edges[edge]
        nweights[edge] = weights[myEdgeID]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#    Functions to perform SL-Linkage on kNN. Functions take a weight matrix
#    and a neighbors matrix. They also take the output matrix in the input.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def knn_slhac(weights, neighbors, Z):
    n_samples, n_neighbors = weights.shape
    
    track = np.arange(n_samples, dtype = np.int32)

    Z_pointer = 0
    while Z_pointer != n_samples - 1:
        a_min = weights.argmin()
        pattern, neigh_idx = a_min // n_neighbors, a_min % n_neighbors
        
        # get neighbor
        neigh = neighbors[pattern, neigh_idx]

        pattern_track = track[pattern]
        neigh_track = track[neigh]        
        
        # unconnected clusters
        # pick any two different clusters and cluster them
        if weights[pattern, neigh_idx] == np.inf:
            clust1 = track[0]
            for i in range(1, n_samples):
                clust2 = track[i]
                if clust1 != clust2:
                    # update the clusters of the samples in track
                    track[track == clust1] = n_samples + Z_pointer
                    track[track == clust2] = n_samples + Z_pointer

                    # add cluster to Z
                    Z[Z_pointer, 0] = pattern_track
                    Z[Z_pointer, 1] = neigh_track
                    Z[Z_pointer, 2] = np.inf
                    Z_pointer += 1
                    break
            continue

        # check if patterns belong to same cluster
        if pattern_track != neigh_track:

            # update the clusters of the samples in track
            track[track == pattern_track] = n_samples + Z_pointer
            track[track == neigh_track] = n_samples + Z_pointer

            # add cluster to Z
            Z[Z_pointer, 0] = pattern_track
            Z[Z_pointer, 1] = neigh_track
            Z[Z_pointer, 2] = weights[pattern, neigh_idx]
            Z_pointer += 1

        # remove distance in coassoc
        weights[pattern, neigh_idx] = np.inf

@jit(nopython=True)
def knn_slhac_fast(weights, neighbors, Z):
    n_samples, n_neighbors = weights.shape
    
    # allocate and fill track array
    # the track array has the current cluster of each pattern
    track = np.empty(n_samples, dtype = np.int32)
    for i in range(n_samples):
        track[i] = i

    Z_pointer = 0
    while Z_pointer != n_samples - 1:
        # get the index of the minimum value in the weights matrix
        a_min = weights.argmin()
        pattern = a_min // n_neighbors
        neigh_idx = a_min % n_neighbors

        # get neighbor corresponding to the neighbor index
        neigh = neighbors[pattern, neigh_idx]

        # get clusters of origin and destination
        pattern_track = track[pattern]
        neigh_track = track[neigh]        
        
        # weight = inf means there are no connected patterns
        # pick any two different clusters and cluster them
        if weights[pattern, neigh_idx] == np.inf:
            clust1 = track[0]
            for i in range(1, n_samples):
                clust2 = track[i]
                if clust1 != clust2:
                    new_clust = n_samples + Z_pointer
                    # update the clusters of the samples in track
                    for i in range(n_samples):
                        i_clust = track[i]
                        if i_clust == pattern_track or i_clust == neigh_track:
                            track[i] = new_clust

                    # add cluster to solution
                    Z[Z_pointer, 0] = pattern_track
                    Z[Z_pointer, 1] = neigh_track
                    Z[Z_pointer, 2] = np.inf
                    Z_pointer += 1
                    break
            continue

        # check if patterns belong to same cluster
        if pattern_track != neigh_track:

            new_clust = n_samples + Z_pointer
            # update the clusters of the samples in track
            for i in range(n_samples):
                i_clust = track[i]
                if i_clust == pattern_track or i_clust == neigh_track:
                    track[i] = new_clust

            # add cluster to solution
            Z[Z_pointer, 0] = pattern_track
            Z[Z_pointer, 1] = neigh_track
            Z[Z_pointer, 2] = weights[pattern, neigh_idx]
            Z_pointer += 1

        # remove distance in coassoc
        weights[pattern, neigh_idx] = np.inf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#    Generic SL-Linkage
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def scipy_numba_slink_wraper(weights, n):
    hie_eac.dists_dtype = weights.dtype
    Z = np.empty((n-1,4), dtype=np.float32)
    hie_eac.slink(weights, Z, n)
    return Z


def slhac(weights, Z):
    """
    modified from knn_slhac.
    """
    n_samples, n_neighbors = weights.shape
    max_val = np.iinfo(weights.dtype).max
    
    track = np.arange(n_samples, dtype = np.int32)

    Z_pointer = 0
    while Z_pointer != n_samples - 1:
        a_min = weights.argmin()
        pattern, neigh_idx = a_min // n_neighbors, a_min % n_neighbors
        
        # get neighbor
        #neigh = neighbors[pattern, neigh_idx]
        neigh = neigh_idx # redundant naming to keep code

        pattern_track = track[pattern]
        neigh_track = track[neigh]        
        
        # unconnected clusters
        # pick any two different clusters and cluster them
        if weights[pattern, neigh_idx] == max_val:
            clust1 = track[0]
            for i in range(1, n_samples):
                clust2 = track[i]
                if clust1 != clust2:
                    # update the clusters of the samples in track
                    track[track == clust1] = n_samples + Z_pointer
                    track[track == clust2] = n_samples + Z_pointer

                    # add cluster to Z
                    Z[Z_pointer, 0] = pattern_track
                    Z[Z_pointer, 1] = neigh_track
                    Z[Z_pointer, 2] = max_val
                    Z_pointer += 1
                    break
            continue

        # check if patterns belong to same cluster
        if pattern_track != neigh_track:

            # update the clusters of the samples in track
            track[track == pattern_track] = n_samples + Z_pointer
            track[track == neigh_track] = n_samples + Z_pointer

            # add cluster to Z
            Z[Z_pointer, 0] = pattern_track
            Z[Z_pointer, 1] = neigh_track
            Z[Z_pointer, 2] = weights[pattern, neigh_idx]
            Z_pointer += 1

        # remove distance in coassoc
        weights[pattern, neigh_idx] = max_val

@jit(nopython=True)
def slhac_fast(weights, Z, max_val):
    n_samples, n_neighbors = weights.shape
    
    # allocate and fill track array
    # the track array has the current cluster of each pattern
    track = np.empty(n_samples, dtype = np.int32)
    for i in range(n_samples):
        track[i] = i

    Z_pointer = 0
    while Z_pointer != n_samples - 1:
        # get the index of the minimum value in the weights matrix
        a_min = weights.argmin()
        pattern = a_min // n_neighbors
        neigh_idx = a_min % n_neighbors

        # get neighbor corresponding to the neighbor index
        #neigh = neighbors[pattern, neigh_idx]
        neigh = neigh_idx

        # get clusters of origin and destination
        pattern_track = track[pattern]
        neigh_track = track[neigh]        
        
        # weight = max_val means there are no connected patterns
        # pick any two different clusters and cluster them
        if weights[pattern, neigh_idx] == max_val:
            clust1 = track[0]
            for i in range(1, n_samples):
                clust2 = track[i]
                if clust1 != clust2:
                    new_clust = n_samples + Z_pointer
                    # update the clusters of the samples in track
                    for i in range(n_samples):
                        i_clust = track[i]
                        if i_clust == pattern_track or i_clust == neigh_track:
                            track[i] = new_clust

                    # add cluster to solution
                    Z[Z_pointer, 0] = pattern_track
                    Z[Z_pointer, 1] = neigh_track
                    Z[Z_pointer, 2] = max_val
                    Z_pointer += 1
                    break
            continue

        # check if patterns belong to same cluster
        if pattern_track != neigh_track:

            new_clust = n_samples + Z_pointer
            # update the clusters of the samples in track
            for i in range(n_samples):
                i_clust = track[i]
                if i_clust == pattern_track or i_clust == neigh_track:
                    track[i] = new_clust

            # add cluster to solution
            Z[Z_pointer, 0] = pattern_track
            Z[Z_pointer, 1] = neigh_track
            Z[Z_pointer, 2] = weights[pattern, neigh_idx]
            Z_pointer += 1

        # remove distance in coassoc
        weights[pattern, neigh_idx] = max_val

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#    Get final clustering from linkage matrix
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def labels_from_Z(Z, n_clusters):
    n_samples = Z.shape[0] + 1
    
    track = np.arange(n_samples, dtype = np.int32)
    
    Z_pointer = 0
    while Z_pointer != n_samples - n_clusters:
        clust0 = Z[Z_pointer, 0]
        clust1 = Z[Z_pointer, 1]
        
        # update the clusters of the samples in track
        track[track == clust0] = n_samples + Z_pointer
        track[track == clust1] = n_samples + Z_pointer

        Z_pointer += 1

    # rename labels
    i=0
    for l in np.unique(track):
        track[track == l] = i
        i += 1
        
    return track

@jit(nopython=True)
def labels_from_Z_numba(Z, track, n_clusters):
    # track is an array of size n_samples
    n_samples = track.size

    for i in range(n_samples):
        track[i] = i

    Z_pointer = 0
    while Z_pointer != n_samples - n_clusters:
        clust0 = Z[Z_pointer, 0]
        clust1 = Z[Z_pointer, 1]
        

        # update the clusters of the samples in track
        new_clust = n_samples + Z_pointer
        for i in range(n_samples):
            curr_track = track[i]
            if curr_track == clust0 or curr_track == clust1:
                track[i] = new_clust

        Z_pointer += 1

    map_key = np.empty(n_clusters, np.int32)
    map_val = np.empty(n_clusters, np.int32)

    for i in range(n_clusters):
        map_key[i] = -1
        map_val[i] = i

    for l in range(n_samples):
        clust = track[l]

        # search for clust in map
        key = -1
        found = 0
        for k in range(n_clusters):
            if map_key[k] == clust:
                found = 1
                key = k
                break
            elif map_key[k] == -1:
                key = k
                break

        # if not found, add clust to map
        if found == 0:
            map_key[key] = clust

        val = map_val[key]

        track[l] = val

@jit(nopython=True)
def binary_search(key, ary):
    """
    Inputs:
        key         : value to find
        ary         : sorted arry in which to find the key

    """
    imin = 0
    imax = ary.size

    while imin < imax:
        imid = (imax + imin) / 2
        imid_val = ary[imid]

        # key is before
        if key < imid_val:
            imax = imid
        # key is after
        elif key > imid_val:
            imin = imid + 1
        # key is between first edge of imid and next first edge
        else:
            return imid
    return -1