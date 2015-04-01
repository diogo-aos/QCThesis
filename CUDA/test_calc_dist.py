# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:32:02 2015

@author: Diogo Silva
"""

import numpy as np
from K_Means import *
from sklearn import datasets # generate gaussian mixture
from timeit import default_timer as timer # timing

class testAttributes:
    dist_mat = None
    assign = None
    groupedData = None
    computedCentroids = None

##generate data
n = 2e6
d = 2
k = 20

n = np.int(n)

total_bytes = np.float((n * d + k * d + n * k) * 4)
print 'Memory used by arrays:\t',total_bytes/1024,'\tKBytes'
print '\t\t\t',total_bytes/(1024*1024),'\tMBytes'

print 'Memory used by data:  \t',n * d * 4 / 1024,'\t','KBytes'

## Generate data
#data = np.random.random((n,d)).astype(np.float32)
data, groundTruth = datasets.make_blobs(n_samples=n,n_features=d,centers=k,
                                        center_box=(-1000.0,1000.0))
data = data.astype(np.float32)

att_np = testAttributes()
att_cu_man = testAttributes()
att_cu_auto = testAttributes()

grouper = K_Means(N=n,D=d,K=k)
centroids = grouper._init_centroids(data)

times = dict()


# Distance matrix
start = timer()
att_np.dist_mat = grouper._np_calc_dists(data,centroids)
times["dist_mat np"] = timer() - start

start = timer()
att_cu_man.dist_mat = grouper._cu_calc_dists(data,centroids,gridDim=None,
                                     blockDim=None,memManage='auto',
                                     keepDataRef=False)
times["dist_mat cuda manual"] = timer() - start

start = timer()
att_cu_auto.dist_mat  = grouper._cu_calc_dists(data,centroids,gridDim=None,
                                     blockDim=None,memManage='auto',
                                     keepDataRef=False)
times["dist_mat cuda auto"] = timer() - start


print "Distance matrix"
print "Numpy == CUDA Man:    ",'\t', np.allclose(att_np.dist_mat,
                                                 att_cu_man.dist_mat)
print "Numpy == CUDA Auto:   ",'\t', np.allclose(att_np.dist_mat,
                                                 att_cu_auto.dist_mat)
print "CUDA Auto == CUDA Man:",'\t', np.allclose(att_cu_auto.dist_mat,
                                                 att_cu_man.dist_mat)

# Assignment and grouped data
start = timer()
att_np.assign,att_np.groupedData = grouper._assign_data(data,att_np.dist_mat)
times["assign and group"] = timer() - start


# Centroid calculation
start = timer()
att_np.computedCentroids = grouper._np_recompute_centroids(att_np.groupedData)
times["centroid computation"] = timer() - start

print att_np.computedCentroids

print "Times"
for k,v in times.iteritems():
    print k,': ',v