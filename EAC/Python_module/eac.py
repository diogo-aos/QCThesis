import numpy as numpy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


	"""
		all features of Matlab toolbox + not full matrix
	"""

	def __init__(self,nsamples):
		self._N = nsamples
		self._assoc_mode = None
		self._prot_mode = None
	def fit(self,ensemble,files=False,assoc_mode="full",prot_mode="random",nprot=None):
		"""
		partitions 		: list of partitions; each partition is a list of arrays (clusterings);
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


		self._coassoc = self._create_coassoc(mode,self._N,nprot=nprot)

		if mode is not "full":
			self._build_prototypes(nprot=nprot,mode=mode)

		for partition in ensemble:
			for clustering in partition:
				self._update_coassoc_matrix(clustering)

		pass

	def _create_coassoc(self,mode,nsamples,nprot=None):

		if mode == "full":
			coassoc = np.empty((nsamples,nsamples))
		elif mode =="prot":
			if nprot == None:
				nprot = np.sqrt(nsamples)
			coassoc = np.empty((nsamples,nprot))
		else:
			validValues=("full","prot")
			raise ValueError("mode value should be from the list:\t" + str(validValues))

		return coassoc


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

	def _build_prototypes(self,nprot=None,mode="random"):
		if nprot = None:
			nprot = np.sqrt(self._N)

		if mode == "random":
			self.k_labels = self._build_random_prototypes(nprot,self._N)
		elif mode == "knn":
			raise Exception("_build_prototypes mode Not implemented.")
		elif mode == "other":
			raise Exception("_build_prototypes mode Not implemented.")
		else:
			validValues=("random","knn","other")
			raise ValueError("Mode value should be from the list:\t" + str(validValues))
		

	def _build_random_prototypes(self,nprot,nsamples):
		# select nprot random samples from the dataset
		return np.random.randint(0,nsamples,nprot)

	def _build_knn_prototypes(self,nprot,data):
		# K-Nearest Neighbours algorithm
		# should return an NxK array with the labels
		pass

	def _build_k_prototypes(self,nprot,nsamples):
		# K-Means / K-Medoids algorithm
		# should return a N-length array with he indices of the chosen data
		pass


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
		k_labels is an array of length K where the k-th element is the index of a data point 
		that corresponds to the k-th prototype.
		"""

		nclusters = len(clusters)
		for i in xrange(nclusters):

			if clusters[i].shape > 1:

				# all data points in cluster - rows to select
				n_in_cluster = clusters[i]

				# all prototypes in cluster - columns to select
				k_in_cluster = np.where(np.in1d(n_in_cluster,k_labels))

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
					k_in_cluster = np.where(np.in1d(n_in_cluster,k_neighbours[j]))

					# this indexing selects the rows and columns specified by n_in_cluster and k_in_cluster
					assoc_mat[j,k_in_cluster] += 1 # np.newaxis is alias for None

			
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
