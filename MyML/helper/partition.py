# -*- coding: utf-8 -*-
"""
Created on 07-05-2015

@author: Diogo Silva

TODO:

"""

import numpy as np

def convertIndexToBin(clusts=None,n_clusts=None,N=None):
	"""
	Converts partition in list of arrays (one array per cluster) format to binary matrix.
	"""

	# clusts is a list of numpy.arrays where each element in
	# in the array is the index of a sample that belongs to that cluster
	
	if clusts == None:
		raise Exception("A clustering partition must be provided.")
	
	if N == None:
		N = 0
		for c in clusts:
			N += c.size

	if n_clusts == None:
		n_clusts=len(clusts)

	clust_out=np.zeros((n_clusts,N))

	for i,clust in enumerate(clusts):
		clust_out[i,clust] = 1
		# for j,ind in enumerate(clust):
		# 	clust_out[i,j]=1

	return clust_out

def convertClusterStringToBin(clusts,n_clusts=None,N=None):
	"""
	Converts partition in array format to binary matrix.

	Converts N length array where the i-th element contains the id of the cluster that the
	i-th samples belongs too to a CxN binary matrix where each row corresponds to a cluster
	and the j-th column of the i-th row is 1 iff the j-th samples belongs to the i-th column.

	In the case that cluster ID can be zero then there is an offset of -1 in the rows, e.g.
	the C-th row actually corresponds to the first cluster.

	clusts 		: N length array with the cluster labels of the N samples
	n_clusts 	: number of clusters; optional
	N 			: number of samples; optional
	"""
	if clusts == None:
		raise Exception("A clustering partition must be provided.")

	if N == None:
		N=clusts.shape[0]

	if n_clusts == None:
		n_clusts=np.max(clusts)

	if np.min(clusts) == 0:
		n_clusts += 1

	clust_out=np.zeros((n_clusts,N))

	for sample_ind,clust_ind in enumerate(clusts):
		# cluster_ind is never 0 so we need to subtract 1 to index the array
		clust_out[clust_ind-1,sample_ind] = 1

	return clust_out


def convertClusterStringToIndex(partition):
	"""
	Converts a partition in the string format (array where the i-th value
	is the cluster label of the i-th pattern) to index format (list of arrays,
	there the k-th array contains the pattern indices that belong to the k-th cluster)
	"""
	clusters=np.unique(partition)
	nclusters=clusters.size

	finalPartition = [None] * nclusters
	for c,l in enumerate(clusters):
		finalPartition[c] = np.where(partition==l)[0]

	return finalPartition

def generateEnsemble(data,generator,n_clusters=20,npartitions=30,iters=3):
	"""
	TODO: check if generator has fit method and n_clusters,labels_ attributes
	"""
	ensemble = [None]*npartitions


	if type(n_clusters) is list:
		clusterRange = True
		min_ncluster=n_clusters[0]
		max_ncluster=n_clusters[1]
	else:
		clusterRange = False
		generator.n_clusters=n_clusters

	generator.max_iter = iters

	for x in xrange(npartitions):
		if clusterRange:
			k = np.random.randint(min_ncluster,max_ncluster)
			generator.n_clusters=k

		generator.fit(data)
		ensemble[x] = convertClusterStringToIndex(generator.labels_)

	return ensemble