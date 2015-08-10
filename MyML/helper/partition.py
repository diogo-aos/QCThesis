# -*- coding: utf-8 -*-
"""
Created on 07-05-2015

@author: Diogo Silva

TODO:

"""

import numpy as np
import matplotlib.pyplot as plt
import os

def convertIndexToBin(clusts=None,n_clusts=None,N=None):
    """
    Converts partition in list of arrays (one array per cluster) format to binary matrix.
    """

    # clusts is a list of numpy.arrays where each element in
    # in the array is the index of a sample that belongs to that cluster
    
    if clusts is None:
        raise Exception("A clustering partition must be provided.")
    
    if N is None:
        N = 0
        for c in clusts:
            N += c.size

    if n_clusts == None:
        n_clusts=len(clusts)

    clust_out=np.zeros((n_clusts,N))

    for i,clust in enumerate(clusts):
        clust_out[i,clust] = 1
        # for j,ind in enumerate(clust):
        #     clust_out[i,j]=1

    return clust_out

def convertClusterStringToBin(clusts, n_clusts=None, N=None):
    """
    Converts partition in array format to binary matrix.

    Converts N length array where the i-th element contains the id of the cluster that the
    i-th samples belongs too to a CxN binary matrix where each row corresponds to a cluster
    and the j-th column of the i-th row is 1 iff the j-th samples belongs to the i-th column.

    In the case that cluster ID can be zero then there is an offset of -1 in the rows, e.g.
    the C-th row actually corresponds to the first cluster.

    clusts         : N length array with the cluster labels of the N samples
    n_clusts     : number of clusters; optional
    N             : number of samples; optional
    """
    if clusts is None:
        raise Exception("A clustering partition must be provided.")

    if N is None:
        N = clusts.shape[0]

    if n_clusts is None:
        n_clusts = np.max(clusts)

    if np.min(clusts) == 0:
        n_clusts += 1

    clust_out = np.zeros((n_clusts,N))

    for sample_ind, clust_ind in enumerate(clusts):
        # cluster_ind is never 0 so we need to subtract 1 to index the array
        clust_out[clust_ind-1, sample_ind] = 1

    return clust_out


def convertClusterStringToIndex(partition):
    """
    Converts a partition in the string format (array where the i-th value
    is the cluster label of the i-th pattern) to index format (list of arrays,
    there the k-th array contains the pattern indices that belong to the k-th cluster)
    """
    clusters = np.unique(partition)
    nclusters = clusters.size
    # nclusters = partition.max() # for cluster id = 0, 1, 2, 3, ....

    finalPartition = [None] * nclusters
    for c,l in enumerate(clusters):
        finalPartition[c] = np.where(partition==l)[0].astype(np.int32)

    return finalPartition

def generateEnsemble(data, generator, n_clusters=20, npartitions=30, iters=3):
    """
    TODO: check if generator has fit method and n_clusters,labels_ attributes
    """
    ensemble = [None] * npartitions

    if type(n_clusters) is list:
        if n_clusters[0] == n_clusters[1]:
            clusterRange = False
            generator.n_clusters = n_clusters[0]
        else:           
            clusterRange = True
            min_ncluster = n_clusters[0]
            max_ncluster = n_clusters[1]
    else:
        clusterRange = False
        generator.n_clusters = n_clusters

    generator.max_iter = iters

    for x in xrange(npartitions):
        if clusterRange:
            k = np.random.randint(min_ncluster, max_ncluster)
            generator.n_clusters = k

        generator.fit(data)
        ensemble[x] = convertClusterStringToIndex(generator.labels_)

    return ensemble


def generateEnsembleToFiles(foldername, data, generator, n_clusters=20,
                            npartitions=30, iters=3, fileprefix="",
                            format_str='%d'):
    """
    TODO: check if generator has fit method and n_clusters,labels_ attributes
    """

    if type(n_clusters) is list:
        if n_clusters[0] == n_clusters[1]:
            clusterRange = False
            generator.n_clusters = n_clusters[0]
        else:           
            clusterRange = True
            min_ncluster = n_clusters[0]
            max_ncluster = n_clusters[1]
    else:
        clusterRange = False
        generator.n_clusters = n_clusters

    generator.max_iter = iters

    for x in xrange(npartitions):
        if clusterRange:
            k = np.random.randint(min_ncluster,max_ncluster)
            generator.n_clusters = k

        generator.fit(data)
        partition = convertClusterStringToIndex(generator.labels_)
        savePartitionToFile(foldername + fileprefix + "part{}.csv".format(x), partition, format_str)


def savePartitionToFile(filename, partition, format_str='%d'):
    """
    Assumes partition as list of arrays.
    """
    n_clusters = len(partition)
    with open(filename, "w") as pfile:
	    for c in xrange(n_clusters):
	        cluster_str = ','.join([format_str % sample for sample in partition[c]])
	        pfile.writelines(cluster_str + '\n')

def loadEnsembleFromFiles(filelist = None, foldername = None):
	if filelist is None and foldername is None:
		raise Exception("EITHER FILELIST OR FOLDERNAME MUST BE SUPPLIED")
	if filelist is None:		
		filelist = [os.path.join(root, name)
		                   for root, dirs, files in os.walk(foldername)
		                   for name in files
		                   if "part" in name]	
	ensemble = list()
	for filename in filelist:
		ensemble.append(loadPartitionFromFile(filename))
	return ensemble

def loadPartitionFromFile(filename):
	partition = list()
	with open(filename, "r") as pfile:
		for pline in pfile:
			if pline != '':
				partition.append(np.fromstring(pline, dtype=np.int32, sep=','))

	return partition


class PlotEnsemble:
    def __init__(self, ensemble, data):
        self.ensemble = ensemble
        self.data = data
        self.curr_partition = 0
        self.n_partitions = len(ensemble)

    def plot(self, num = None, draw_perimeter=False):
        if num is None:
            self._plotPartition(self.curr_partition, draw_perimeter)
            if self.curr_partition < self.n_partitions - 1:
                self.curr_partition += 1
            else:
                self.curr_partition = 0
        elif num >= self.n_partitions or num < 0:
            raise Exception("Invalid partition index.")
        else:
            self.curr_partition = num
            self._plotPartition(self.curr_partition, draw_perimeter)

    def _plotPartition(self, clust_idx, draw_perimeter=False):
        
        if not draw_perimeter:
            for clust in self.ensemble[clust_idx]:
                plt.plot(self.data[clust, 0], self.data[clust, 1], '.')

        else:
            from scipy.spatial import ConvexHull
            for clust in self.ensemble[clust_idx]:
                points = self.data[clust]
                hull = ConvexHull(points)
                
                plt.plot(points[:,0], points[:,1], 'o')
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        plt.title("Partition #{}, Num. clusters = {}".format(clust_idx,
            len(self.ensemble[clust_idx])))

    def maxClusterSize(self):
        max_size = 0
        for part in self.ensemble:
            for clust in part:
                clust_len = len(clust)
                if clust_len > max_size:
                    max_size = clust_len
        return max_size


