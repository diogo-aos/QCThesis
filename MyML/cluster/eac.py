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
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
from .K_Means3 import K_Means
from random import sample

class EAC():

	def __init__(self,nsamples,data=None):
		self._N = nsamples
		self._assoc_mode = None
		self._prot_mode = None

		self.data = data
		self.npartitions = 0

	def fit(self,ensemble,files=False,assoc_mode="full",prot_mode="random",
			nprot=None,link='single',build_only=False):
		"""
		ensemble 		: list of partitions; each partition is a list of arrays (clusterings);
						  each array contains the indices of the cluster's data;  if files=True,
						  partitions is a list of file names, each corresponding to a partition
		assoc_mode 		: type of association matrix; "full" - NxN, "prot" - NxK prototypes
		prot_mode 		: how to build the prototypes; "random" - random selection of K data points,
						  "knn" for K-nearest neighbours, "other" for K centroids/medoids
		nprot 			: number of prototypes to use; default = sqrt(number of samples)
		"""
		# how to build association matrix
		if self._assoc_mode is None:
			self._assoc_mode = assoc_mode
		# how to build prototypes
		if self._prot_mode is None:
			self._prot_mode = prot_mode

		# create co-association matrix
		self._coassoc = self._create_coassoc(assoc_mode,self._N,nprot=nprot)

		if assoc_mode is not "full":
			# changing assoc_mode for the matrix updates
			if prot_mode is "knn":
				self._assoc_mode="knn"
			else:
				self._assoc_mode="other"


			self._build_prototypes(nprot=nprot,mode=prot_mode, data=self.data)

		# received names of partition files
		if files:
			for partition_file in ensemble:
				partition = self._readPartition(partition_file) # read partition from file
				self._update_coassoc_matrix(partition) # update co-association matrix
		# received partitions
		else:
			for partition in ensemble:
				self._update_coassoc_matrix(partition) # update co-association matrix

		# transform votes in dissimilarity matrix
		self._diss_assoc = self._coassoc / self.npartitions # normalize
		self._diss_assoc = self._diss_assoc.max() - self._diss_assoc # transform in dissimilarity

		# apply linkage
		if assoc_mode is "full" and not build_only:
		
			self.hierarchy = self._apply_linkage(self._diss_assoc,link)

		pass

	def _create_coassoc(self,mode,nsamples,nprot=None):

		if mode == "full":
			coassoc = np.zeros((nsamples,nsamples))
		elif mode =="prot":
			if nprot == None:
				nprot = np.sqrt(nsamples)
			coassoc = np.zeros((nsamples,nprot))
		else:
			validValues=("full","prot")
			raise ValueError("mode value should be from the list:\t" + str(validValues))

		return coassoc

	def _readPartition(self,filename):
		# list to hold the cluster arrays
		partition = list()

		with open(filename,"r") as pfile:
			# read cluster lines
			for cluster_line in pfile:
				if cluster_line == '\n':
					continue
				cluster = np.fromstring(cluster_line,sep=',',dtype=np.int32)
				partition.append(cluster)

		return partition

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # #                                BUILD PROTOTYPES                                   # # # #
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

	def _build_prototypes(self,nprot=None,mode="random",data=None):
		if nprot == None:
			nprot = np.sqrt(self._N)

		if mode == "random":
			self.k_labels = self._build_random_prototypes(nprot,self._N)

		elif mode == "knn":
			if data is None:
				raise Exception("Data needs to be set for this method of choosing prototypes.")
			self.k_neighbours = self._build_knn_prototypes(nprot,data)

		elif mode == "other":
			if data is None:
				raise Exception("Data needs to be set for this method of choosing prototypes.")
			self.k_labels = self._build_k_prototypes(nprot,data)

		else:
			validValues=("random","knn","other")
			raise ValueError("Mode value should be from the list:\t" + str(validValues))
		

	def _build_random_prototypes(self,nprot,nsamples):

		# select nprot unique random samples from the dataset
		return sample(xrange(nsamples),nprot)

	def _build_knn_prototypes(self,nprot,data):
		# K-Nearest Neighbours algorithm
		# should return an NxK array with the labels
		nneigh=nprot+1 #first neighbour is the point itself, it gets discarded afterwards

		# Minkowski distance is a generalization of Euclidean distance and is equivelent to it for p=2
		neigh = NearestNeighbors(n_neighbors=nneigh, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2)
		neigh.fit(data)

		k_indices = neigh.kneighbors(X=data,return_distance=False)
		k_indices = k_indices[:,1:] # discard first neighbour

		return k_indices

	def _build_k_prototypes(self,nprot,data):
		# K-Means / K-Medoids algorithm
		# should return a N-length array with he indices of the chosen data
	    grouper = K_Means()
	    grouper._centroid_mode = "index"
	    grouper.fit(data, nprot, iters=300, mode="cuda", cuda_mem='manual',tol=1e-4,max_iters=300)
	    centroids = grouper.centroids

	    nclusters = centroids.shape[0]

	    # TODO - very inefficient
	    k_labels = np.zeros(nclusters,dtype=np.int32)

	    for k in xrange(nclusters):
	    	dist = data - centroids[k]
	    	dist = dist ** 2
	    	dist = dist.sum(axis=1)

	    	k_labels[k] = dist.argmin()
		return k_labels


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # #                               UPDATE CO-ASSOCIATION MATRIX                        # # # #
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

	def _update_coassoc_matrix(self,clusters):
		"""
		clusters 				: 	list of arrays; each array with the indices (int) of the
									samples in the corresponding cluster
		"""

		# full matrix
		if self._assoc_mode is "full":
			self._update_coassoc_n(self._coassoc,clusters)
		# reduced matrix
		elif self._assoc_mode is "other":
			self._update_coassoc_k(self._coassoc,clusters,self.k_labels)
		elif self._assoc_mode is "knn":
			self._update_coassoc_knn(self._coassoc,clusters,self.k_neighbours)
		else:
			validValues=("full","knn","other")
			raise ValueError("mode value should be from the list:\t" + str(validValues))

		# increment number of partitions (# times updated)
		self.npartitions += 1


	def _update_coassoc_n(self,assoc_mat,clusters):
		"""
		Updates a square NxN co-association matrix.
		"""
		nclusters = len(clusters)
		for i in xrange(nclusters):
			if clusters[i].shape > 1:

				n_in_cluster = clusters[i] # n_in_cluster = indices of samples in cluster

				# this indexing selects the rows and columns specified in sic
				assoc_mat[n_in_cluster[:,np.newaxis],n_in_cluster] += 1

	def _update_coassoc_k(self,assoc_mat,clusters,k_labels):
		"""
		Updates an NxK co-association matrix.
		k_labels is an array (List, not np.ndarray) of length K where the k-th element is the index of a data point 
		that corresponds to the k-th prototype.
		"""

		nclusters = len(clusters)
		for i in xrange(nclusters):

			if clusters[i].shape > 1:

				# all data points in cluster - rows to select
				n_in_cluster = clusters[i]

				## all prototypes in cluster - columns to select
				# in1d checks common values between two 1-D arrays (a,b) and returns boolean array
				# with the shape of a with value True on the indices of common values
				k_in_cluster = np.where(np.in1d(k_labels,n_in_cluster))[0]

				if k_in_cluster.size == 0:
					continue
				

				# this indexing selects the rows and columns specified by n_in_cluster and k_in_cluster
				assoc_mat[n_in_cluster[:,np.newaxis],k_in_cluster] += 1 # np.newaxis is alias for None


	def _update_coassoc_knn(self,assoc_mat,clusters,k_neighbours):
		"""
		Updates an NxK co-association matrix.
		k_neighbours is an NxK array where the k-th element of the i-th row is the index of a data point 
		that corresponds to the k-th nearest neighbour of the i-th data point. That neighbour is the k-th
		prototype of the i-th data point.
		"""
		nclusters = len(clusters)
		for i in xrange(nclusters):

			if clusters[i].shape > 1:

				# all data points in cluster - rows to select
				n_in_cluster = clusters[i]

				# update row j of matrix
				for j in n_in_cluster:
					# all prototypes in cluster - columns to select
					k_in_cluster = np.where(np.in1d(k_neighbours[j],n_in_cluster))[0]

					if k_in_cluster.size == 0:
						continue

					# this indexing selects the rows and columns specified by n_in_cluster and k_in_cluster
					assoc_mat[j,k_in_cluster] += 1 # np.newaxis is alias for None
		pass

	def _update_coassoc_knn_sparse(assoc_mat,clusters,k_neighbours):
	    """
	    Updates an NxK co-association matrix.
	    k_neighbours is an NxK array where the k-th element of the i-th row is the index of a data point 
	    that corresponds to the k-th nearest neighbour of the i-th data point. That neighbour is the k-th
	    prototype of the i-th data point.
	    """
	    nclusters = len(clusters)
	    for i in xrange(nclusters):

	        if clusters[i].shape > 1:

	            # all data points in cluster - rows to select
	            n_in_cluster = clusters[i]

	            # update row j of matrix
	            for j in n_in_cluster:
	                # all prototypes in cluster - columns to select
	                # column indices corresponding to the K-prototypes	                
	                k_in_cluster = n_in_cluster[np.in1d(n_in_cluster,k_neighbours[j])]
	                
	                # this indexing selects the rows and columns specified by n_in_cluster and k_in_cluster
	                #assoc_mat[j,k_in_cluster] += np.ones_like(k_in_cluster)
	                assoc_mat[k_in_cluster[:,np.newaxis],k_in_cluster] += np.ones_like((k_in_cluster.size,k_in_cluster.size))
	        pass
			
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # #                               FINAL CLUSTERING                                    # # # #
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # #                                                                                   # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

	def _apply_linkage(self,assoc_mat,method='single'):
		"""
		SciPy linkage wants a distance array of format pdist. SciPy squareform 
		converts between the two formats.

		assoc_mat 	: pair-wise similarity association matrix
		method 		: linkage method to use; can be 'single'(default), 'complete',
					  'average', 'weighted', 'centroid', 'median', 'ward'
		"""

		condensed_assoc = squareform(assoc_mat)

		# convert pair-wise similarity array (assoc_mat->condensed_assoc) to dissimilarity
		condensed_diss_assoc = condensed_assoc.max() - condensed_assoc

		Z = linkage(condensed_diss_assoc,method=method)

		return Z