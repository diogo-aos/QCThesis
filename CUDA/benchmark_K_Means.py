# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:59:23 2015

@author: Diogo Silva


Testbench for K_Means


# TODO:
- create main loops
- design and create datastructures
- create datasets
- run algorithms
- print feedback (what is running, ETA, etc.)
"""

import numpy as np
import K_Means3
from K_Means3 import *
from sklearn import datasets # generate gaussian mixture
from timeit import default_timer as timer # timing


# Setup logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a file handler
handler = logging.FileHandler('benchmark_K_Means.log')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

# create a console handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(consoleHandler)

logger.info('Start of logging.')

# datasets configs to use - program will iterate over each combination of 
# parameters:
# - cadinality - number of points to use
# - dimensionality - number of dimensions
# - clusters . number of clusters to use
# - rounds - number of rounds to repeat tests
# - iters - number of iterations or convergence

cardinality = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 2e6, 4e6]
dimensionality = [2]
nat_clusters = [20]
clusters = [10, 20, 30, 100, 500]
rounds = 10 
iterations=[3]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            HELPER FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def generateData(n,d,k):
    n_int = np.int(n)

    # Generate data
    data, groundTruth = datasets.make_blobs(n_samples=n_int,n_features=d,centers=k,
                                            center_box=(-1000.0,1000.0))
    data = data.astype(np.float32)  
    
    return data

import pickle

def readPickle(filename):
    """
    Receives a filename to read from, opens the file, loads the pickled data 
    inside to memory and returns an object with that data.
    """

    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data
    
def writePickle(filename,data):
    """
    Receives a filename to write to and the data to write, opens the file, 
    writes the data to the file.
    """
    
    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            CUDA
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

logger.info("CUDA")
bench_results = dict()

# iterate on number of datapoints
for i,n in enumerate(cardinality):

    # iterate on dimension of data
    for i,d in enumerate(dimensionality):

        # iterate on number of natural clusters in mix
        for i,nc in enumerate(nat_clusters):

            log_str = "N={0},D={1},NATC={2}".format(n,d,nc)
            logger.info('Generating data:' + log_str)

            # generate data
            data = generateData(n,d,nc)      
            
            # iterate on number of clusters to find
            for i,k in enumerate(clusters):
                
                # iterate on the iterations to perform
                for i,iters in enumerate(iterations):

                    rounds_times = list()
                    
                    r = 0
                    while r < rounds:
                        log_str = "N={0},D={1},NATC={2},K={3},ITERS={4},ROUND={5}".format(n,d,nc,k,iters,r)
                        logger.info('CUDA clustering:' + log_str)

                        start = timer()
                        grouperCUDA = K_Means()
                        grouperCUDA._centroid_mode="index"
                        grouperCUDA.fit(data, k, iters=iters, mode="cuda", cuda_mem='manual',tol=1e-4,max_iters=300)
                        try:
                            grouperCUDA.fit(data, k, iters=iters, mode="cuda", cuda_mem='manual',tol=1e-4,max_iters=300)
                        except:
                            logger.info('CUDA: BAD ITERATION')
                            continue    
                        runtime = timer() - start
                        logger.info('CUDA time:' + str(runtime))
                        rounds_times.append(runtime)

                        r += 1

                    record = ("cuda",n,d,nc,k,iters)
                    bench_results[record] = rounds_times
                    del grouperCUDA

            del data

writePickle("CUDA_results.pkl",bench_results)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            NUMPY
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

logger.info("NUMPY")
bench_results = dict()

# iterate on number of datapoints
for i,n in enumerate(cardinality):

    # iterate on dimension of data
    for i,d in enumerate(dimensionality):

        # iterate on number of natural clusters in mix
        for i,nc in enumerate(nat_clusters):

            log_str = "N={0},D={1},NATC={2}".format(n,d,nc)
            logger.info('Generating data:' + log_str)

            # generate data
            data = generateData(n,d,nc)      
            
            # iterate on number of clusters to find
            for i,k in enumerate(clusters):
                
                # iterate on the iterations to perform
                for i,iters in enumerate(iterations):

                    rounds_times = list()
                    
                    r = 0
                    while r < rounds:
                        log_str = "N={0},D={1},NATC={2},K={3},ITERS={4},ROUND={5}".format(n,d,nc,k,iters,r)
                        logger.info('NumPy clustering:' + log_str)

                        start = timer()
                        grouperNP = K_Means()
                        grouperNP._centroid_mode="index"
                        try:
                            grouperNP.fit(data, k, iters=iters, mode="numpy", cuda_mem='manual',tol=1e-4,max_iters=300)
                        except:
                            logger.info('NumPy: BAD ITERATION')
                            continue    
                        runtime = timer() - start
                        rounds_times.append(runtime)

                        r += 1

                    record = ("numpy",n,d,nc,k,iters)
                    bench_results[record] = rounds_times
                    del grouperNP

            del data

writePickle("NumPy_results.pkl",bench_results)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            PYTHON
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

logger.info("PYTHON")
bench_results = dict()

# iterate on number of datapoints
for i,n in enumerate(cardinality):

    # iterate on dimension of data
    for i,d in enumerate(dimensionality):

        # iterate on number of natural clusters in mix
        for i,nc in enumerate(nat_clusters):

            log_str = "N={0},D={1},NATC={2}".format(n,d,nc)
            logger.info('Generating data:' + log_str)

            # generate data
            data = generateData(n,d,nc)      
            
            # iterate on number of clusters to find
            for i,k in enumerate(clusters):
                
                # iterate on the iterations to perform
                for i,iters in enumerate(iterations):

                    rounds_times = list()
                    
                    r = 0
                    while r < rounds:
                        log_str = "N={0},D={1},NATC={2},K={3},ITERS={4},ROUND={5}".format(n,d,nc,k,iters,r)
                        logger.info('Python clustering:' + log_str)

                        start = timer()
                        grouperP = K_Means()
                        grouperP._centroid_mode="index"
                        try:
                            grouperP.fit(data, k, iters=iters, mode="python", cuda_mem='manual',tol=1e-4,max_iters=300)
                        except:
                            logger.info('Python: BAD ITERATION')
                            continue    
                        runtime = timer() - start
                        rounds_times.append(runtime)

                        r += 1

                    record = ("numpy",n,d,nc,k,iters)
                    bench_results[record] = rounds_times
                    del grouperP

            del data

writePickle("Python_results.pkl",bench_results)