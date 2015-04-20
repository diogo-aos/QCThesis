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
from K_Mea
from sklearn import datasets # generate gaussian mixture
from timeit import default_timer as timer # timing

import sys

# Setup logging
import logging


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    GLOBAL VARS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

synthetic_data = False
filename_prefix = None
mode = "numpy"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    LOGGING

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Status logging
logger = logging.getLogger('status')
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

# Results logging
resultsLogger = logging.getLogger('results')
resultsLogger.setLevel(logging.INFO)

# create a file handler for results
resultsHandler = logging.FileHandler('results.csv')
resultsHandler.setLevel(logging.INFO)

resultsLogger.addHandler(resultsHandler)


resultsLogger.info('type,N,D,NATC,K,iters,R,time') #csv header
logger.info('Start of logging.')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    ARGUMENTS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="where to read data from",type=str)
parser.add_argument("-n", "--nsamples", help="number of samples for synthetic data",type=int)
parser.add_argument("-D", "--dimension", help="dimension for synthetic data",type=int)
parser.add_argument("-C", "--centers", help="number of clusters for synthetic data",type=int)
parser.add_argument("-c", "--clusters", help="number of clusters",type=int)
parser.add_argument("-i", "--iterations", help="number of iterations",type=int)
parser.add_argument("-m", "--mode", help="computation mode (cuda,numpy,python)",type=string)

args = parser.parse_args()

#########################################################


if args.data == "synthetic":
    synthetic_data = True


# file names
tailString=args.tailString

filenames=dict()
filenames['data']=args.data
filenames['qk'] = args.data[:-4] + '_' + 'qk' + tailString
filenames['k'] = args.data[:-4] + '_' + 'k' + tailString
filenames['params'] = args.data[:-4] + '_' + 'params' + tailString

# Quantum parameters
numClusters=args.clusters
numOracles=args.oracles
qubitStringLen=args.stringLen
qGenerations=args.generations
numRounds=args.rounds

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                    GETTING DATA

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if synthetic_data:
    from sklearn.datasets import make_blobs # generate gaussian mixture

    data, groundTruth = datasets.make_blobs(n_samples=n_int,n_features=d,centers=k,
                                            center_box=(-1000.0,1000.0))


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
clusters = [5, 10, 20, 30, 40, 50, 100, 250, 500]
rounds = 10 
iterations=[3]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            CUDA
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

logger.info("CUDA")

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
                        except KeyboardInterrupt:
                            print "Cleaning up..."
                            del grouperCUDA, data
                            print "Exiting..."
                            sys.exit(0)
                        except:
                            logger.info('CUDA: BAD ITERATION')
                            continue    
                        runtime = timer() - start
                        logger.info('CUDA time:' + str(runtime))
                        rounds_times.append(runtime)

                        # save results
                        result_line = "cuda,{0},{1},{2},{3},{4},{5},{6}".format(n,d,nc,k,iters,r,runtime)
                        resultsLogger.info(result_line)

                        r += 1

                    del grouperCUDA

            del data


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            NUMPY
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

logger.info("NUMPY")

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
                        except KeyboardInterrupt:
                            print "Cleaning up..."
                            del grouperNP, data
                            print "Exiting..."
                            sys.exit(0)                        
                        except:
                            logger.info('NumPy: BAD ITERATION')
                            continue
                        runtime = timer() - start


                        logger.info('NumPy time:' + str(runtime)) # status logging

                        # save results
                        result_line = "numpy,{0},{1},{2},{3},{4},{5},{6}".format(n,d,nc,k,iters,r,runtime)
                        resultsLogger.info(result_line)

                        r += 1

                    del grouperNP

            del data


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                            PYTHON
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

logger.info("PYTHON")

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
                        except KeyboardInterrupt:
                            print "Cleaning up..."
                            del grouperP, data
                            print "Exiting..."
                            sys.exit(0)                          
                        except:
                            logger.info('Python: BAD ITERATION')
                            continue    
                        runtime = timer() - start
                        logger.info('Python time:' + str(runtime))

                        # save results
                        result_line = "python,{0},{1},{2},{3},{4},{5},{6}".format(n,d,nc,k,iters,r,runtime)
                        resultsLogger.info(result_line)

                        r += 1

                    del grouperP

            del data
