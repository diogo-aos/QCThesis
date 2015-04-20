import numpy as numpy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

class EAC():
	"""
		all features of Matlab toolbox + not full matrix
	"""


	def _update_coassoc_matrix(self,assoc_mat,nsamples_in_clusters,clusters):
		"""
		nsamples_in_clusters 	: 	array with the number of samples in each cluster
		clusters 				: 	list of arrays; each array with the indices (int) of the
									samples in the corresponding cluster
		"""

		nclusters = len(clusters)
		for i in xrange(nclusters):
			if nsamples_in_clusters[i] > 1:

				sic = clusters[i] # sic = indices of samples in cluster

				# this indexing selects the rows and columns specified in sic
				assoc_mat[sic[:,np.newaxis],sic] += 1

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
