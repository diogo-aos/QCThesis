import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

import oracle
import qubitLib
import DaviesBouldin

# Receives:
#  - mixture
#  - numClusters
#  - numInits
# Returns:
#  - k_centroids
#  - qk_assignment
#  - k_timings_cg

def k_means(mixture,numClusters,numInits):

	k_timings_cg=list()
	start=datetime.now()

	k_assignment=list()
	k_centroids=list()
	k_inertia=list()

	for i in range(numInits):
		estimator = KMeans(n_clusters=numClusters,init='k-means++',n_init=1)
		assignment = estimator.fit_predict(mixture)
		centroids = estimator.cluster_centers_

		k_centroids.append(centroids)
		k_assignment.append(assignment)
		k_inertia.append(estimator.inertia_)

		k_timings_cg.append((datetime.now() - start).total_seconds())
		start=datetime.now()

	return k_centroids,k_assignment,k_timings_cg
