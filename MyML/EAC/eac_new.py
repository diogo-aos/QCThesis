# -*- coding: utf-8 -*-
"""
Created on 10-04-2015

@author: Diogo Silva

Evidence accumulation clustering. This module aims to include all
features of the Matlab toolbox plus addressing NxK co-association
matrices.

TODO:
- clustering of non-square co-association matrix
- link everything
- add sanity checks on number of samples of partitions
- robust exception handling
- fix centroid-based prototype creation
- convert dissimilarity matrix to float32 (why?)
"""

import numpy as np
from random import sample
from numba import jit, njit
# from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.sparse.csgraph import connected_components

from scipy_numba.sparse.csgraph import minimum_spanning_tree
from scipy_numba.spatial.distance import squareform

from MyML.cluster.linkage import scipy_numba_slink_wraper as slink
from MyML.cluster.linkage import labels_from_Z
from MyML.cluster.K_Means3 import K_Means

from MyML.EAC.sparse import EAC_CSR, _compute_max_assocs_from_ensemble as biggest_cluster_size
from MyML.EAC.full import EAC_FULL

def sqrt_rule(n):
    n_clusters = [np.sqrt(n)/2, np.sqrt(n)]
    n_clusters = map(np.ceil, n_clusters)
    n_clusters = map(int, n_clusters)
    return n_clusters

def sqrt2_rule(n):
    n_clusters = [np.sqrt(n), np.sqrt(n)*2]
    n_clusters = map(np.ceil, n_clusters)
    n_clusters = map(int, n_clusters)
    return n_clusters

class EAC():

    def __init__(self, n_samples, **kwargs):
        """
        mat_sparse         : stores co-associations in a sparse matrix
        mat_half         : stores co-associations in pdist format, in an (n*(n-1))/2 length array
        """

        self.n_samples = n_samples

        # check if all arguments were passed as a dictionary
        args = kwargs.get("args")
        if args is not None and type(args) == dict:
            kwargs == args

        ## generate ensemble parameters
        self.n_partitions = kwargs.get("n_partitions", 100)
        self.iters = kwargs.get("iters", 3)
        self.n_clusters = kwargs.get("n_clusters", "sqrt")
        self.toFiles = False
        self.toFiles_folder = None

        ## build matrix parameters
        self.condensed = kwargs.get("condensed", True)
        self.kNN = kwargs.get("kNN", False)
        self.assoc_dtype = kwargs.get("assoc_dtype", np.uint8)

        # sparse matrix parameters
        self.sparse = kwargs.get("sparse", False)
        self.sp_sort = kwargs.get("sparse_sort_mode", "surgical")
        self.sp_max_assocs = kwargs.get("sparse_max_assocs", None)
        self.sp_max_assocs_factor = kwargs.get("sparse_max_assocs_factor", 3)
        self.sp_max_assocs_mode = kwargs.get("sparse_max_assocs_mode", "linear")
        self.sp_keep_degree = kwargs.get("sparse_keep_degree", False)

        # if not sparse and not kNN then it is full matrix
        if not self.sparse  and not self.kNN:
            self.full = True
        else:
            self.full = False

        ## final clustering parameters
        self.linkage = kwargs.get("linkage", "SL")

    def _validate_params(self):
        pass

    def generateEnsemble(self):
        pass

    def buildMatrix(self, ensemble):

        if self.sparse:
            if self.sp_max_assocs is None:
                self.sp_max_assocs = biggest_cluster_size(ensemble)
                self.sp_max_assocs *= self.sp_max_assocs_factor
            
            coassoc = EAC_CSR(self.n_samples, max_assocs=self.sp_max_assocs,
                              condensed=self.condensed,
                              max_assocs_type=self.sp_max_assocs_mode,
                              sort_mode=self.sp_sort,
                              dtype=self.assoc_dtype)

            coassoc.update_ensemble(ensemble)
            coassoc._condense(keep_degree = self.sp_keep_degree)
        elif self.full:
            coassoc = EAC_FULL(self.n_samples, condensed=self.condensed,
                               dtype=self.assoc_dtype)
            coassoc.update_ensemble(ensemble)
        elif self.kNN:
            raise NotImplementedError("kNN matrix building has not been included in this version.")
        else:
            raise ValueError("Build matrix: No sparse, no full, no kNN. No combination possible.")

        self.coassoc = coassoc

        # received names of partition files
        # if files:
        #     for partition_file in ensemble:
        #         partition = self._readPartition(partition_file) # read partition from file
        #         self._update_coassoc_matrix(partition) # update co-association matrix
        # # received partitions
        # else:
        #     for partition in ensemble:
        #         self._update_coassoc_matrix(partition) # update co-association matrix

    def finalClustering(self, n_clusters=0):
        if self.sparse:
            n_fclusts, labels = sp_sl_lifetime(self.coassoc.csr,
                                               max_val=self.n_partitions,
                                               n_clusters=n_clusters)
        elif self.full:
            n_fclusts, labels = full_sl_lifetime(self.coassoc.coassoc,
                                                 self.n_samples,
                                                 max_val=self.n_partitions,
                                                 n_clusters=n_clusters)
        elif self.kNN:
            raise NotImplementedError("kNN not included in this version yet.")
        else:
            raise ValueError("Final clustering: No sparse, no full, no kNN. No combination possible.")

        self.n_fclusts = n_fclusts
        self.labels = labels
        return labels

    def fit(self, ensemble,files=False, assoc_mode="full", prot_mode="none",
            nprot=None, link='single', build_only=False):
        """
        ensemble    : list of partitions; each partition is a list of 
                      arrays (clusterings); each array contains the indices
                      of the cluster's data;  if files=True, partitions is
                      a list of file names, each corresponding to a partition
        assoc_mode  : type of association matrix; "full" - NxN, "prot" - NxK prototypes
        prot_mode   : how to build the prototypes; "random" - random selection 
                      of K data points, "knn" for K-nearest neighbours, "other"
                      for K centroids/medoids
        nprot       : num. of prototypes to use; default = sqrt(num. of samples)
        """
        # how to build association matrix
        if self._assoc_mode is None:
            self._assoc_mode = assoc_mode
        # how to build prototypes
        if self._prot_mode is None:
            self._prot_mode = prot_mode

        # create co-association matrix
        self._coassoc = self._create_coassoc(assoc_mode, self.n_samples, nprot=nprot)

        if prot_mode is not "none":
            # changing assoc_mode for the matrix updates
            if prot_mode == "knn":
                self._assoc_mode="knn"

            elif assoc_mode == "full" and prot_mode == "random":
                self._assoc_mode = "full_random"   

            elif prot_mode == "random":
                self._assoc_mode = "random"
           
            else:
                self._assoc_mode="other"

            self._build_prototypes(nprot=nprot, mode=prot_mode, data=self.data)

        self.n_partitions = 0



        # delete diagonal
        #self._coassoc[xrange(self.n_samples),xrange(self.n_samples)] = np.zeros(self.n_samples)

        # convert sparse matrix to convenient format, if it is sparse
        if self.mat_sparse:
            self._coassoc = self._coassoc.tocsr()
        # else:
        # 	self._coassoc[np.diag_indices_from(self._coassoc)] = 0

    def _create_coassoc(self, mode, nsamples, nprot=None):
        if self.condensed:
            n = sum(xrange(1, nsamples))
            coassoc = np.zeros(n, dtype=self.assoc_type)
        elif mode == "full":
            if self.mat_sparse:
                coassoc = sparse_type((nsamples, nsamples), dtype=self.assoc_type)
            else:
                coassoc = np.zeros((nsamples, nsamples), dtype=self.assoc_type)
        elif mode =="prot":
            if nprot == None:
                nprot = np.sqrt(nsamples)
            coassoc = np.zeros((nsamples,nprot), dtype=self.assoc_type)
        else:
            validValues=("full", "prot")
            raise ValueError("mode value should be from the list:\t" + str(validValues))

        return coassoc

    def _readPartition(self, filename):
        # list to hold the cluster arrays
        partition = list()

        with open(filename, "r") as pfile:
            # read cluster lines
            for cluster_line in pfile:
                if cluster_line == '\n':
                    continue
                cluster = np.fromstring(cluster_line, sep=',', dtype=np.int32)
                partition.append(cluster)

        return partition


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                      BUILD PROTOTYPES                       # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _build_prototypes(self, nprot=None, mode="random", data=None):
        if nprot == None:
            nprot = np.sqrt(self.n_samples)

        if mode == "random":
            self.k_labels = self._build_random_prototypes(nprot, self.n_samples)
            self.k_labels.sort()

        elif mode == "knn":
            if data is None:
                raise Exception("Data needs to be set for this method of choosing prototypes.")
            self.k_neighbours = self._build_knn_prototypes(nprot, data)

        elif mode == "other":
            if data is None:
                raise Exception("Data needs to be set for this method of choosing prototypes.")
            self.k_labels = self._build_k_prototypes(nprot, data)

        else:
            validValues=("random","knn","other")
            raise ValueError("Mode value should be from the list:\t" + str(validValues))
        

    def _build_random_prototypes(self, nprot, nsamples):

        # select nprot unique random samples from the dataset
        return np.array(sample(xrange(nsamples), nprot), dtype=np.int32)

    def _build_knn_prototypes(self, nprot, data):
        """
        K-Nearest Neighbours algorithm
        should return an NxK array with the labels
        """
        #first neighbour is the point itself, it gets discarded afterwards
        nneigh = nprot + 1 

        # Minkowski distance is a generalization of Euclidean distance and 
        # is equivelent to it for p=2
        neigh = NearestNeighbors(n_neighbors=nneigh, radius=1.0,
                                 algorithm='auto', leaf_size=30,
                                 metric='minkowski', p=2)
        neigh.fit(data)

        k_indices = neigh.kneighbors(X=data, return_distance=False)
        k_indices = k_indices[:,1:] # discard first neighbour

        return k_indices

    def _build_k_prototypes(self, nprot, data):
        # K-Means / K-Medoids algorithm
        # should return a N-length array with he indices of the chosen data
        grouper = K_Means()
        grouper._centroid_mode = "index"
        grouper.fit(data, nprot, iters=300, mode="cuda", cuda_mem='manual',
                    tol=1e-4, max_iters=300)
        centroids = grouper.centroids

        nclusters = centroids.shape[0]

        # TODO - very inefficient
        k_labels = np.zeros(nclusters, dtype=np.int32)

        for k in xrange(nclusters):
            dist = data - centroids[k]
            dist = dist ** 2
            dist = dist.sum(axis=1)

            k_labels[k] = dist.argmin()
        return k_labels


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #              UPDATE CO-ASSOCIATION MATRIX                   # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _update_coassoc_k(self, assoc_mat, clusters, k_labels):
        """
        Updates an NxK co-association matrix.
        k_labels is an array (List, not np.ndarray) of length K where the k-th 
        element is the index of a data point that corresponds to the
        k-th prototype.
        """

        nclusters = len(clusters)
        for i in xrange(nclusters): # for each cluster in ensemble
            # if cluster has more than 1 sample (i.e. not outlier)
            if clusters[i].size > 1: 

                # all data points in cluster - rows to select
                n_in_cluster = clusters[i]

                ## select prototypes present in cluster - columns to select
                # in1d checks common values between two 1-D arrays (a,b) and 
                # returns boolean array with the shape of a with value True on
                # the indices of common values
                k_in_cluster = np.where(np.in1d(k_labels, n_in_cluster))[0]

                if k_in_cluster.size == 0:
                    continue                

                # this indexing selects the rows and columns specified by 
                # n_in_cluster and k_in_cluster; np.newaxis is alias for None
                assoc_mat[n_in_cluster[:, np.newaxis], k_in_cluster] += 1

    def _update_coassoc_full_k(self, assoc_mat, clusters, k_labels):
        """
        Updates an NxN co-association matrix with only K prototypes specified
        by k_labels. k_labels is an array (List, not np.ndarray) of length K 
        where the k-th element is the index of a data point that corresponds
        to the k-th prototype.
        """

        nclusters = len(clusters)
        for i in xrange(nclusters): # for each cluster in ensemble
            # if cluster has more than 1 sample (i.e. not outlier)
            if clusters[i].size > 1: 

                # all data points in cluster - rows to select
                n_in_cluster = clusters[i]

                ## select prototypes present in cluster - columns to select
                # in1d checks common values between two 1-D arrays (a,b) and 
                # returns boolean array with the shape of a with value True on
                # the indices of common values
                prots_in_cluster = np.intersect1d(k_labels, n_in_cluster)
                points_in_cluster = np.setdiff1d(n_in_cluster, prots_in_cluster)

                if prots_in_cluster.size == 0:
                    continue                

                # this indexing selects the rows and columns specified by 
                # n_in_cluster and k_in_cluster; np.newaxis is alias for None
                # select all rows that are not prots and all columns that are 
                # prots and increment them
                assoc_mat[points_in_cluster[:, np.newaxis], prots_in_cluster] += 1
                assoc_mat[prots_in_cluster[:, np.newaxis], points_in_cluster] += 1

                # select all rows and columns that are prots and increment them
                assoc_mat[prots_in_cluster[:, np.newaxis], prots_in_cluster] += 1

                
    def _update_coassoc_knn(self, assoc_mat, clusters, k_neighbours):
        """
        Updates an NxK co-association matrix.
        k_neighbours is an NxK array where the k-th element of the i-th row is
        the index of a data point that corresponds to the k-th nearest neighbour
        of the i-th data point. That neighbour is the k-th prototype of the 
        i-th data point.
        """
        nclusters = len(clusters)
        for i in xrange(nclusters):

            if clusters[i].size > 1:

                # all data points in cluster - rows to select
                n_in_cluster = clusters[i]

                # update row j of matrix
                for j in n_in_cluster:
                    # all prototypes in cluster - columns to select
                    k_in_cluster = np.in1d(k_neighbours[j], n_in_cluster)

                    if k_in_cluster.size == 0:
                        continue

                    # this indexing selects the rows and columns specified by
                    # n_in_cluster and k_in_cluster
                    assoc_mat[j, k_in_cluster] += 1 # newaxis is alias for None
        pass
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                      OPERATIONS                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # #                                                             # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def apply_threshold(self, threshold):
        """
        threshold   : all co-associations whose value is below 
                      threshold * max_val are zeroed
        max_val     : usually number of partitions
        assoc_mat   : co-association matrix
        """
        assoc_mat = self._coassoc
        max_val = self.n_partitions
        apply_threshold_to_coassoc(threshold, max_val, assoc_mat)

    def getMaxAssocs(self):
        """\
        Returns the maximum number of co-associations a sample has and the
        index of that sample.\
        """ 
        # if not hasattr(self, 'degree'):
        #     self._getAssocsDegree()
        # return self.degree.max()


        if not self.mat_sparse:
            max_assocs, max_idx = get_max_assocs_in_sample(self._coassoc)
        else:
            max_assocs, max_idx = get_max_assocs_in_sample_csr(self._coassoc)
        return max_assocs, max_idx

    def _getAssocsDegree(self):
        self.degree = np.zeros(self.n_samples, dtype=np.int32)
        if self.condensed:
            error_str = "Getting degree from condensed matrix. Alternative: " +\
                         "convert to 2d, get degree, multiply by 2."
            raise NotImplementedError(error_str)
        elif not self.mat_sparse:
            full_get_assoc_degree(self._coassoc, self.degree)
        else:
            self.degree = self._coassoc.indptr[1:] - self._coassoc.indptr[:-1]
        self.nnz = self.degree.sum()


    def getNNZAssocs(self):
        """Get total number of associations in co-association matrix."""
        if not self.mat_sparse:
            #return np.count_nonzero(self.._coassoc)
            return numba_array2d_nnz(self._coassoc, self._coassoc.shape[0],
                                     self._coassoc.shape[1])
        else:
            return self._coassoc.getnnz()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # #                                                             # # # #
# # # #                                                             # # # #
# # # #                                                             # # # #
# # # #                      FINAL CLUSTERING                       # # # #
# # # #                                                             # # # #
# # # #                                                             # # # #
# # # #                                                             # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




def sp_sl_lifetime(mat, max_val=False, n_clusters=0):
    """
    Converts graph weights to dissimilarities if input graph is in 
    similarities. Computes MST (Kruskal) of dissimilarity graph.
    Compute number of disconnected clusters (components).
    Sort MST in increasing order to get equivalent of SL clustering.
    Compute lifetimes if number of clusters is not provided.
    Make necessary cuts to have the desired number of clusters.
    Compute connected components (clusters) after the cuts.

    Inputs:
        graph           : dis/similarity matrix in CS form.
        max_val         : maximum value from which dissimilarity will be
                          computed. If False (default) assumes input graph
                          already encodes dissimilarities.
        n_clusters      : number of clusters to compute. If 0 (default), 
                          use lifetime criteria.
    Outputs:
        n_fclusts       : final number of clusters        
        labels          : final clustering labels
    """

    dtype = mat.dtype

    # converting to diassociations
    if max_val != False:
        mat.data = max_val + 1 - mat.data

    # get minimum spanning tree
    mst = minimum_spanning_tree(mat)

    # compute number of disconnected components
    n_disconnect_clusters = mst.shape[0] - mst.nnz

    # sort associations by weights
    asort = mst.data.argsort()
    sorted_weights = mst.data[asort]

    if n_clusters == 0:
        cont, max_lifetime = lifetime_n_clusters(sorted_weights)

        if n_disconnect_clusters > 1:
            # add 1 to max_val as the maximum weight because I also added
            # 1 when converting to diassoc to avoid having zero weights
            disconnect_lifetime = max_val + 1 - sorted_weights[-1]

            # add disconnected clusters to number of clusters if disconnected
            # lifetime is smaller
            if max_lifetime > disconnect_lifetime:
                cont += n_disconnect_clusters - 1
            else:
                cont = n_disconnect_clusters

        nc_stable = cont
    else:
        nc_stable = n_clusters

    # cut associations if necessary
    if nc_stable > n_disconnect_clusters:
        n_cuts = nc_stable - n_disconnect_clusters
        
        mst.data[asort[-n_cuts:]] = 0
        mst.eliminate_zeros()   

    if nc_stable > 1:
        n_comps, labels = connected_components(mst)
    else:
        labels = np.empty(0, dtype=np.int32)
        n_comps = 1  

    return n_comps, labels



def full_sl_lifetime(mat, n_samples, max_val=False, n_clusters=0):

    dtype = mat.dtype

    # convert to diassoc
    if mat.ndim == 2:
        mat = squareform(mat)

    # converting to diassociations
    if max_val != False:
        make_diassoc_1d(mat, max_val + 1)

    #Z = linkage(mat, method="single")
    Z = slink(mat, n_samples)

    if n_clusters == 0:

        cont, max_lifetime = lifetime_n_clusters(Z[:,2])

        nc_stable = cont
    else:
        nc_stable = n_clusters

    if nc_stable > 1:
        labels = labels_from_Z(Z, n_clusters=nc_stable)
        # rename labels
        i=0
        for l in np.unique(labels):
            labels[labels == l] = i
            i += 1        
    else:
        labels = np.empty(0, dtype=np.int32)

    return nc_stable, labels

def lifetime_n_clusters(weights):
    # compute lifetimes
    lifetimes = weights[1:] - weights[:-1]

    # maximum lifetime
    m_index = np.argmax(lifetimes)
    th = lifetimes[m_index]

    # get number of clusters from lifetimes
    indices = np.where(weights >weights[m_index])[0]
    if indices.size == 0:
        cont = 1
    else:
        cont = indices.size + 1

    #testing the situation when only 1 cluster is present
    # if maximum lifetime is smaller than 2 times the minimum
    # don't make any cuts (= 1 cluster)
    # max>2*min_interval -> nc=1
    close_to_zero_indices = np.where(np.isclose(lifetimes, 0))
    minimum = np.min(lifetimes[close_to_zero_indices])

    if th < 2 * minimum:
        cont = 1

    return cont, th

# 2d
@njit
def make_diassoc_2d(ary, val):
    for row in range(ary.shape[0]):
        for col in range(ary.shape[1]):
            tmp = ary[row,col]
            ary[row,col] = val - tmp

#1d
@njit
def make_diassoc_1d(ary, val):
    for i in range(ary.size):
        tmp = ary[i]
        ary[i] = val - tmp

