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
    nclusters=np.unique(partition).shape[0]
    finalPartition=[list() for x in xrange(nclusters)]
    for n,c in partition:
        finalPartition[c].append(n)
    
    for c in xrange(len(finalPartition)):
    	finalPartition[c] = np.array(finalPartition[c])

    return finalPartition