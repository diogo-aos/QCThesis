import numpy as numpy

class EAC():
	"""
		all features of matlab toolbox + not full matrix
	"""


	def _update_coassoc_matrix(assoc_mat, nsamples_in_clusters,clusters):
		"""
		nsamples_in_clusters 	: 	array with the number of samples in each cluster
		clusters 				: 	list os arrays. Each array with the indices of the samples in 
									the corresponding cluster
		"""

		nclusters = len(clusters)
		for i in xrange(nclusters):
			if nsamples_in_clusters[i] > 1:
				sic = clusters[i] #sic = samples in cluster
				assoc_mat[sic,sic] += 1


	def