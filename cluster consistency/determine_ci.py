# -*- coding: utf-8 -*-
"""
Created on 

@author: Diogo Silva

TODO:
- save number of elements per cluster somewhere to use for B and C in match,
  instead of performing the computations
- mode where CxN matrix is not built, instead every cluster is converted to binary every time
- don't use matrix multiplication to check shared (benchmark performance)
"""

import numpy as np

class ConsistencyIndex:

	def __init__(self,N=None):
		self.N = N

	def score(self,clusts1,clusts2,format='array',N=None):
		"""
		clusts1,clusts2 	: the two partitions to match
		format 				: format of partitions: 'array' or 'list';
			'array'			: partition is an
							  array with the length of the number of samples
							  where the i-th row has the cluster the i-th samples
							  belongs to;
			'list'			: partition is a list of arrays, the k-th array
							  has the indices of the samples that belong to it.

		"""

		if format=='list':
			clusts1_=self._convertIndexToPos(clusts=clusts1,N=self.N)
			clusts2_=self._convertIndexToPos(clusts=clusts2,N=self.N)
		elif format=='array':
			clusts1_=self._convertClusterStringToPos(clusts=clusts1,N=self.N)
			clusts2_=self._convertClusterStringToPos(clusts=clusts2,N=self.N)
		else:
			raise Exception("Format not accepted.")

		self.clusts1_=clusts1_
		self.clusts2_=clusts2_

		self._match_(clusts1_,clusts2_,N=self.N)

		self.ci = self.match_count / self.N

		return self.ci


	def _match_(self,clusts1,clusts2,n_clusts1=None,n_clusts2=None,N=None):


		# these copies will be altered
		clusts1_ = clusts1
		clusts2_ = clusts2

		if n_clusts1 is None:
			n_clusts1 = clusts1_.shape[0]

		if n_clusts2 is None:
			n_clusts2 = clusts2_.shape[0]

		# total shared samples between clusters of both partitions
		n_shared = 0

		for it in range(np.min([n_clusts1,n_clusts2])):

			#compute best match between all clusters of both partitions
			max_coef=0
			k,l = -1,-1
			savedA = -1
			for i in range(n_clusts1):
				for j in range(n_clusts2):
					## compute match coefficient

					# number of matches between cluster i of partition 1 and j of 2
					A = clusts1_[i,:].dot(clusts2_[j,:])

					# number of samples in clust 1
					B = clusts1_[i,:].dot(clusts1_[i,:])
					#B = clusts1_[i,:].nonzero()

					# number of samples in clust 2
					C = clusts2_[j,:].dot(clusts2_[j,:])
					#C = clusts2_[i,:].nonzero()

					match_coef = A / (B + C - A)

					if match_coef > max_coef:
						k,l = i,j
						savedA = A

			# increment shared samples
			n_shared += savedA

			# delete clusters (rows) from partitions
			clusts1 = np.delete(clusts1,k,axis=0)
			clusts2 = np.delete(clusts2,l,axis=0)

			# decrement number of clusters
			n_clusts1 -= 1
			n_clusts2 -= 1

		self.match_count = n_shared
		self.unmatch_count = self.N - n_shared

		return n_shared

	def _convertIndexToPos(self,clusts=None,n_clusts=None,N=None):
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
			for j,ind in enumerate(clust):
				clust_out[i,j]=1

		return clust_out

	def _convertClusterStringToPos(self,clusts=None,n_clusts=None,N=None):
		if clusts == None:
			raise Exception("A clustering partition must be provided.")

		if N == None:
			N=clusts.shape[0]

		if n_clusts == None:
			n_clusts=max(clusts)

		clust_out=np.zeros((n_clusts,N))

		for sample_ind,clust_ind in enumerate(clusts):
			# cluster_ind is never 0 so we need to subtract 1 to index the array
			clust_out[clust_ind-1,sample_ind] = 1

		return clust_out

"""
class HIndex:

	def __init__(self):
	

"""