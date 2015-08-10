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
import hungarian
from ..helper.partition import convertIndexToBin as convertIndexToPos
from ..helper.partition import convertClusterStringToBin as convertClusterStringToPos

class ConsistencyIndex:

	def __init__(self,N=None):
		self.N = N

	def score(self, clusts1, clusts2,format='array',format1=None,format2=None,array_has_zero=False,N=None):
		"""
		clusts1,clusts2 	: the two partitions to match
		format 				: format of partitions: 'array' or 'list';
			'array'			: partition is an
							  array with the length of the number of samples
							  where the i-th row has the cluster the i-th samples
							  belongs to;
			'list'			: partition is a list of arrays, the k-th array
							  has the indices of the samples that belong to it.
		format1/2 			: format of partition 1; superseeds the format parameter
		"""

		if format1 == None:
			format1 = format

		if format2 == None:
			format2 = format

		if format1=='list':
			clusts1_=convertIndexToPos(clusts=clusts1,N=self.N)
		elif format1=='array':
			clusts1_=convertClusterStringToPos(clusts=clusts1,N=self.N)
		else:
			raise Exception("Format not accepted: {}".format(format1))


		if format2=='list':
			clusts2_=convertIndexToPos(clusts=clusts2,N=self.N)
		elif format2=='array':
			clusts2_=convertClusterStringToPos(clusts=clusts2,N=self.N)
		else:
			raise Exception("Format not accepted: {}".format(format2))


		self.clusts1_ = clusts1_
		self.clusts2_ = clusts2_

		self._match_(clusts1_,clusts2_,N=self.N)

		self.ci = np.float(self.match_count) / self.N

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

		for it in xrange(np.min([n_clusts1,n_clusts2])):

			#compute best match between all clusters of both partitions
			max_coef=0
			k,l = -1,-1 # cluster indices of best match
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
						max_coef = match_coef

			# increment shared samples
			n_shared += savedA

			# delete clusters (rows) from partitions
			clusts1_ = np.delete(clusts1_,k,axis=0)
			clusts2_ = np.delete(clusts2_,l,axis=0)
			# decrement number of clusters
			n_clusts1 -= 1
			n_clusts2 -= 1

		self.match_count = n_shared
		self.unmatch_count = self.N - n_shared

		return n_shared


class HungarianIndex():

	def __init__(self,nsamples=None):


		self.N = nsamples

	def score(self, clusts1, clusts2, format='array', format1=None, format2=None):

		if format1 == None:
			format1 = format

		if format2 == None:
			format2 = format

		if format1=='list':
			clusts1_=convertIndexToPos(clusts=clusts1,N=self.N)
		elif format1=='array':
			clusts1_=convertClusterStringToPos(clusts=clusts1,N=self.N)
		elif format1 == 'bin':
			clusts1_ = clusts1
		else:
			raise ValueError("Format not accepted: {}".format(format1))


		if format2=='list':
			clusts2_=convertIndexToPos(clusts=clusts2, N=self.N)
		elif format2=='array':
			clusts2_=convertClusterStringToPos(clusts=clusts2, N=self.N)
		elif format2 == 'bin':
			clusts2_ = clusts2			
		else:
			raise ValueError("Format not accepted: {}".format(format2))

		nclusts1_ = clusts1_.shape[0]
		nclusts2_ = clusts2_.shape[0]

		match_matrix = clusts1_.dot(clusts2_.T)

		self.clusts1_=clusts1_
		self.clusts2_=clusts2_

		# cost/profit matrix is automatically padded to be square with this module
		hungAcc = hungarian.Hungarian(match_matrix, is_profit_matrix=True)
		hungAcc.calculate()

		self.match_count = hungAcc.get_total_potential()
		self.accuracy = np.float(self.match_count) / self.N