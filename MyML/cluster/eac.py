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

from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.neighbors import NearestNeighbors
from random import sample

from scipy.sparse import lil_matrix, csr_matrix, dok_matrix

from MyML.cluster.linkage import slhac_fast, labels_from_Z


from numba import jit

sparse_type = lil_matrix

class EAC():

    def __init__(self, n_samples, data=None, mat_sparse=False, mat_half=False):
        """
        mat_sparse         : stores co-associations in a sparse matrix
        mat_half         : stores co-associations in pdist format, in an (n*(n-1))/2 length array
        """
        
        self.n_samples = n_samples
        self._assoc_mode = None
        self._prot_mode = None

        self.data = data
        self.n_partitions = 0

        # properties of co-association matrix
        self.mat_sparse = mat_sparse
        self.mat_half = mat_half

        self.assoc_type = np.uint8

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

        # received names of partition files
        if files:
            for partition_file in ensemble:
                partition = self._readPartition(partition_file) # read partition from file
                self._update_coassoc_matrix(partition) # update co-association matrix
        # received partitions
        else:
            for partition in ensemble:
                self._update_coassoc_matrix(partition) # update co-association matrix

        # delete diagonal
        #self._coassoc[xrange(self.n_samples),xrange(self.n_samples)] = np.zeros(self.n_samples)

        # convert sparse matrix to convenient format, if it is sparse
        if self.mat_sparse:
            self._coassoc = self._coassoc.tocsr()
        # else:
        # 	self._coassoc[np.diag_indices_from(self._coassoc)] = 0

    def _create_coassoc(self, mode, nsamples, nprot=None):

        if mode == "full":
            
            if self.mat_sparse:
                coassoc = sparse_type((nsamples, nsamples), dtype=self.assoc_type)
            else:
                coassoc = np.zeros((nsamples, nsamples), dtype=self.assoc_type)
        elif mode =="prot":
            if nprot == None:
                nprot = np.sqrt(nsamples)
            coassoc = np.zeros((nsamples,nprot), dtype=np.float32)
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

    def _update_coassoc_matrix(self, clusters):
        """
        clusters    : list of arrays; each array with the indices (int) of the
                      samples in the corresponding cluster
        """

        #print "updating partition {}".format(self.n_partitions)

        # full matrix
        if self._assoc_mode is "full":
            if self.mat_sparse:
                self._update_coassoc_n_sparse(self._coassoc, clusters)
            else:
                #self._update_coassoc_n(self._coassoc, clusters)
                update_coassoc_with_partition(self._coassoc, clusters)

        # reduced matrix
        elif self._assoc_mode is "random":
            self._update_coassoc_k(self._coassoc, clusters, self.k_labels)

        elif self._assoc_mode is "full_random":
            self._update_coassoc_full_k(self._coassoc, clusters, self.k_labels)
            #update_coassoc_with_partition(self._coassoc, clusters, self.k_labels)

        elif self._assoc_mode is "knn":
            self._update_coassoc_knn(self._coassoc, clusters, self.k_neighbours)

        else:
            validValues = ("full", "knn", "other")
            raise ValueError("mode value should be from the list:\t" + str(validValues))

        # increment number of partitions (# times updated)
        self.n_partitions += 1

    def _update_coassoc_n(self, assoc_mat, clusters):
        """
        Updates a square NxN co-association matrix.
        """
        nclusters = len(clusters)
        for i in xrange(nclusters):
            if clusters[i].size > 1:

                n_in_cluster = clusters[i] # n_in_cluster = indices of samples in cluster

                # this indexing selects the rows and columns specified in sic
                #assoc_mat[n_in_cluster[:,np.newaxis],n_in_cluster] += 1
                assoc_mat[n_in_cluster[:, np.newaxis], n_in_cluster] += 1

    def _update_coassoc_n_sparse(self, assoc_mat, clusters):
        """
        Updates a square NxN co-association matrix.
        """
        nclusters = len(clusters)
        for i in xrange(nclusters):
            if clusters[i].size > 1:

                n_in_cluster = clusters[i] # n_in_cluster = indices of samples in cluster

                # this indexing selects the rows and columns specified in sic
                #assoc_mat[n_in_cluster[:,np.newaxis],n_in_cluster] += 1
                for row in n_in_cluster:
                    assoc_mat[row, n_in_cluster] += np.ones(n_in_cluster.size)

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
        if not self.mat_sparse:
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


    def _lifetime_clustering(self, assoc_mat=None, method='single',n_clusters = 0):
        if assoc_mat is None:
            assoc_mat = self._coassoc
        Z = self._apply_linkage(assoc_mat=assoc_mat, method=method)
        labels = self._clusterFromLinkage(Z = Z, n_clusters = n_clusters)
        return labels


    def _apply_linkage(self, assoc_mat=None, method='single'):
        """
        SciPy linkage wants a distance array of format pdist. SciPy squareform 
        converts between the two formats.

        assoc_mat  : pair-wise similarity association matrix
        method     : linkage method to use; can be 'single'(default), 'complete',
                      'average', 'weighted', 'centroid', 'median', 'ward'
        """

        if assoc_mat is None:
            assoc_mat = self._coassoc

        # this should go after the squareform since there is only the upper
        # triangle to compute
        # diassoc = (self.n_partitions - assoc_mat)#/self.n_partitions

        # diassoc[np.diag_indices_from(diassoc)] = 0
        # condensed_diassoc = squareform(diassoc)
        # Z = linkage(condensed_diassoc, method=method)
        
        make_diassoc(assoc_mat, self.n_partitions) # make matrix diassoc
        fill_diag(assoc_mat, 0) # clear diagonal

        condensed_diassoc = squareform(assoc_mat)

        Z = linkage(condensed_diassoc, method=method)


        # Z = np.empty((n_samples-1,3), dtype=np.float32) # allocate Z
        # slhac_fast(assoc_mat, Z) # apply linkage

        self._Z = Z
        return Z

    def _clusterFromLinkage(self, Z=None, n_clusters=0):
        """
        Finds the cluster of highest lifetime. Computes the number of clusters
        according to highest lifetime. Determines the clusters form dendrogram.
        """

        if Z is None:
            Z = self._Z

        if n_clusters == 0:
            # lifetime is here computed as the distance difference between 
            # any two consecutive nodes, i.e. the distance between passing
            # from n to n-1 clusters

            lifetimes = Z[1:,2] - Z[:-1,2]

            m_index = np.argmax(lifetimes)

            # Z is ordered in increasing order by weight of connection
            # the connection whose weight is higher than the one specified
            # by m_index MUST be the one from which the jump originated the
            # maximum lifetime; all connections above that (and including)
            # will be removed for the final clustering
            indices = np.where(Z[:,2] > Z[m_index, 2])[0]
            #indices = np.arange(m_index+1, Z.shape[0])
            if indices.size == 0:
                cont = 1
            else:
                cont = indices.size + 1

            # store maximum lifetime
            th = lifetimes[m_index]

            #testing the situation when only 1 cluster is present
            # if maximum lifetime is smaller than 2 times the minimum
            # don't make any cuts (= 1 cluster)
            #max>2*min_interval -> nc=1
            close_to_zero_indices = np.where(np.isclose(lifetimes, 0))
            minimum = np.min(lifetimes[close_to_zero_indices])

            if th < 2 * minimum:
                cont = 1

            nc_stable = cont

        else:
            nc_stable = n_clusters

        if nc_stable > 1:
            # only the labels are of interest

            labels = labels_from_Z(Z, n_clusters=nc_stable)

            # rename labels
            i=0
            for l in np.unique(labels):
                labels[labels == l] = i
                i += 1
        else:
            labels = np.zeros(self.n_samples, dtype = np.int32)

        self.labels_ = labels
        return labels


#--------------- / ----------------------------------------------------

@jit(nopython=True)
def fill_diag(ary, val):
    for i in range(ary.shape[0]):
        ary[i,i] = val

@jit(nopython=True)
def make_diassoc(ary, val):
    for row in range(ary.shape[0]):
        for col in range(ary.shape[1]):
            tmp = ary[row,col]
            ary[row,col] = val - tmp

def apply_threshold_to_coassoc(threshold, max_val, assoc_mat):
    """
    threshold   : all co-associations whose value is below 
                  threshold * max_val are zeroed
    max_val     : usually number of partitions
    assoc_mat   : co-association matrix
    """
    assoc_mat[assoc_mat < threshold * max_val] = 0

def get_max_assocs_in_sample(assoc_mat):
    """
    Returns the maximum number of co-associations a sample has and the index of
    that sample.
    """
    max_row_size=0
    max_row_idx=-1
    row_idx=0
    for row in assoc_mat:
        if row.nonzero()[0].size > max_row_size:
            max_row_size = row.nonzero()[0].size
            max_row_idx = row_idx
        row_idx += 1
        
    return max_row_size, max_row_idx

def get_max_assocs_in_sample_csr(assoc_mat):
    """
    Returns the maximum number of co-associations a sample has and the index of
    that sample.
    """

    first_col = assoc_mat.indptr

    n_cols = first_col[1:] - first_col[:-1]
    max_row_size = n_cols.max()
    max_row_idx = n_cols.argmax()
    return max_row_size, max_row_idx

# # FULL MATRIX FUNCTIONS
def update_coassoc_with_ensemble(coassoc, ensemble, k_labels = None):
    for p in xrange(len(ensemble)):
        update_coassoc_with_partition(coassoc, ensemble[p], k_labels = k_labels)

def update_coassoc_with_partition(coassoc, partition, k_labels = None):
    for c in xrange(len(partition)):
        if k_labels is None:
            numba_update_coassoc_with_cluster(coassoc, partition[c])
        else:
            numba_update_full_k(coassoc, partition[c], k_labels)

@jit(nopython=True)
def numba_update_coassoc_with_cluster(coassoc, cluster):
    """
    Receives the coassoc 2-d array and the cluster 1-d array. 
    """
    for i in range(cluster.size):
        curr_i = cluster[i]
        for j in range(i+1, cluster.size):
            curr_j = cluster[j]
            if i == j:
                continue
            coassoc[curr_i, curr_j] += 1
            coassoc[curr_j, curr_i] += 1

@jit
def numba_update_full_k_prots(assoc_mat, cluster, k_labels):
    max_prot_size = np.min(cluster.size, k_labels.size)
    prots_in_cluster = np.empty(max_prot_size, dtype=np.int32)
    num_prots_in_cluster = 0

    points_in_cluster = np.empty_like(cluster)
    num_points_in_cluster = 0

    # get list of prototypes in cluster and list of non-prots in cluster
    for s in range(cluster.size):
        s_not_prot = True
        sample = cluster[s]
        for p in range(k_labels.size):
            if k_labels[p] == sample:
                prots_in_cluster[num_prots_in_cluster] = k_labels[p]
                num_prots_in_cluster += 1
                s_not_prot = False
        if s_not_prot:
            points_in_cluster[num_points_in_cluster] = sample
            num_points_in_cluster += 1            

    # fill all prototype cols and rows with non-prot points
    for p in range(num_prots_in_cluster):
        prot = prots_in_cluster[p]
        for s in range(num_points_in_cluster):
            sample = points_in_cluster[s]
            assoc_mat[sample,prot] += 1
            assoc_mat[prot,sample] += 1

    # fill all prototype cols and rows with prot points
    for po in range(num_prots_in_cluster):
        prot_outer = prots_in_cluster[po]
        for pi in range(po, num_prots_in_cluster):
            prot_inner = prots_in_cluster[pi]
            assoc_mat[prot_outer, prot_inner] += 1
            assoc_mat[prot_inner, prot_outer] += 1


    # firstProt = 1
    # for i in range(k_labels.size):# fr each prototype
    #     prot = k_labels[i]
    #     for j in range(cluster.size): # check if prototype is in cluster
    #         firstProt += 1
    #         # if prototype is in cluster
    #         if prot == cluster[j]:
    #             # fill prototype col and row for each point
    #             for k in range(cluster.size):
    #                 point = cluster[k]
    #                 # don't update diagonal
    #                 if prot != point:
    #                     assoc_mat[prot, point] += 1
    #                     assoc_mat[point, prot] += 1
    #                 if point in k_labels:
    #                     pass

    #             # continue to search for next prototypes
    #             continue


@jit
def update_knn_coassoc_with_cluster(coassoc, cluster, neighbours):
    for j in range(cluster.size):
        j_id = cluster[j]

        # check if neighbours of j are in cluster
        for n in range(neighbours.shape[1]):
            if binary_search(neighbours[n], cluster) != -1:
                pass # FINISH

@jit(nopython=True)
def numba_update_coassoc_with_cluster_pdist(coassoc, cluster):
    """
    Receives the coassoc 2-d array and the cluster 1-d array. 
    """
    raise Exception("PDIST FORMAT NOT IMPLEMENTED.")
    for i in range(cluster.size):
        curr_i = cluster[i]
        for j in range(i+1, cluster.size):
            curr_j = cluster[j]
            if i == j:
                continue
            # swap i with j
            if curr_i < curr_j:
                temp = curr_i
                curr_i = curr_j
                curr_j = temp
            
            coassoc[curr_i, curr_j] += 1

@jit(nopython=True)
def numba_array2d_nnz(ary, width, height):
    nnz = 0
    for line in range(height):
        for col in range(width):
            if ary[line,col] != 0:
                nnz = nnz + 1
    return nnz

@jit(nopython=True)
def full_get_assoc_degree(coassoc, degree):
    rows, cols = coassoc.shape
    for row in range(rows):
        for col in range(cols):
            if coassoc[row,col] != 0:
                degree[row] += 1