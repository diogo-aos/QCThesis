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
from K_Means2 import *
from sklearn import datasets # generate gaussian mixture
from timeit import default_timer as timer # timing


# Setup logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('benchmark_K_Means.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.info('Start of logging.')

# datasets configs to use - program will iterate over each combination of 
# parameters:
# - cadinality - number of points to use
# - dimensionality - number of dimensions
# - clusters . number of clusters to use
# - rounds - number of rounds to repeat tests
# - iters - number of iterations of convergence

cardinality = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 2e6, 4e6]
dimensionality = [2]
clusters = [10, 20, 30, 100, 500]
rounds = 10 
iters=[3]

# Setting up datastrutures
bench_results = dict()
bench_resutlts['cuda']=list()


def generateData(n,d):
    # Generate data
    data, groundTruth = datasets.make_blobs(n_samples=n,n_features=d,centers=k,
                                            center_box=(-1000.0,1000.0))
    data = data.astype(np.float32)  
    
    return data
    
def runCUDA(data,k,iters):
    # setup    
    grouperCUDA = K_Means()
    grouperCUDA._cuda_mem = "manual"
    
    # cluster
    start = timer()    
    grouperCUDA.fit(data,k,iters=iters,cuda=True)
    time = timer() - start
    
    return time

def runNP(data,k,iters):
    # setup    
    grouperCUDA = K_Means()
    grouperCUDA._cuda_mem = "manual"
    
    # cluster
    start = timer()    
    grouperCUDA.fit(data,k,iters=iters,cuda=True)
    time = timer() - start
    
    return time    

for i,n in enumerate(cardinality):
    for i,d in enumerate(dimensionality):
        
        # generate data
        data = generateData(n,d,k)      
        
        for i,k in enumerate(clusters): 
            
            for r in xrange(rounds):
                
                #cuda
                start = timer()
                grouperCUDA = K_Means()
                grouperCUDA._cuda_mem = "manual"
                grouperCUDA.fit(data,k,iters=3,mode="cuda")
                times['cuda'] = timer() - start
                
                #numpy                
                start = timer()
                grouperNP = K_Means()
                grouperNP.fit(data,k,mode="numpy")
                times['numpy'] = timer() - start
                
                #python
                start = timer()
                grouperP = K_Means()
                grouperP.fit(data,k,mode="python")
                times['numpy'] = timer() - start
                

# Testing CUDA


# Testing NumPy


# Testing Python