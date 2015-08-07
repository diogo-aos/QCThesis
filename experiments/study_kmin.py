
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd


# In[2]:

import MyML.helper.partition as part
import MyML.cluster.eac as eac
import MyML.cluster.K_Means3 as myKM
import MyML.metrics.accuracy as accuracy
import MyML.utils.profiling as myProf
import MyML.utils.sparse as mySparse
max_assocs_fn = mySparse._compute_max_assocs_from_ensemble

# Setup logging
import logging

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="where to read data from",type=str)
parser.add_argument('-y', "--yes", help="don't ask confirmation of folder",
					action='store_true')
args = parser.parse_args()

folder = args.folder

if not args.yes:
	raw_input("Folder: {}\nIs this correct?".format(folder))
else:
	print "Folder being used is: {}".format(folder)


# Status logging
logger = logging.getLogger('status')
logger.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a file handler
handler = logging.FileHandler(folder + 'study_kmin.log')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

# create a console handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(consoleHandler)

# # bulk study

# ## rules

# In[119]:

# rules for picking kmin kmax 
def rule1(n):
    """sqrt"""
    k = [np.sqrt(n)/2, np.sqrt(n)]
    k = map(np.ceil,k)
    k = map(int, k)
    return k

def rule2(n):
    """2sqrt"""
    k =  map(lambda x:x*2,rule1(n))
    return k

def rule3(n, sk, th):
    """fixed s/k"""
    k = [n * 1.0 / sk, th * n * 1.0 / sk]
    k = map(np.ceil,k)
    k = map(int, k)
    return k

def rule4(n):
    """sk=sqrt/2,th=30%"""
    return rule3(n, sk1(n), 1.3)

def rule5(n):
    """sk=300,th=30%"""
    return rule3(n,300, 1.3)

# rules for picking number of samples per cluster
def sk1(n):
    """sqrt/2"""
    return int(np.sqrt(n) / 2)
    
rules = [rule1, rule2, rule4, rule5]


# ## set-up

# In[212]:




# In[ ]:

logger.info("Loading dataset...")

data = np.genfromtxt(folder + "data.csv", delimiter=',', dtype=np.float32)
gt = np.genfromtxt(folder + "gt.csv", delimiter=',', dtype=np.int32)


# In[120]:

mem_full_max = 20 * 2**30 # max mem full mat can take

cardinality = [1e2,2.5e2,5e2,7.5e2,
               1e3,2.5e3,5e3,7.5e3,
               1e4,2.5e4,5e4,7.5e4,
               1e5,2.5e5,5e5,7.5e5,
               1e6,2.5e6]
cardinality = map(int,cardinality)

total_n = data.shape[0]
div = map(lambda n: total_n / n, cardinality)

rounds = 5
res_lines = rounds * len(cardinality) * len(rules)
res_cols = ['n_samples', 'rule', 'kmin', 'kmax', 't_ensemble',
            'biggest_cluster', 'type_mat', 't_build', 'n_assocs',
            'min_assoc', 'max_assoc', 'mean_assoc', 'std_assoc',
            't_sl', 'accuracy', 'round']
results = pd.DataFrame(index=range(res_lines), columns=res_cols)

t = myProf.Timer() # timer

# ensemble properties
n_partitions = 100
n_iters = 3

# EAC properties
assoc_mode = "full"
prot_mode = "none"

# ## run

logger.info("Starting experiment...")

# In[198]:

res_idx = 0
for d in div: # for each size of dataset
   
    # sample data
    data_sampled = np.ascontiguousarray(data[::d])
    #gt_sampled = gt[::d]
    n = data_sampled.shape[0]

    # if n >= 150000:
    #     break

    # pick sparse on full matrix
    if n **2 < mem_full_max:
        mat_sparse = False
    else:
        mat_sparse = True

    for rule in rules: # for each kmin rule
        n_clusts = rule(n)

        logger.info("Sampled of {} patterns.".format(n))
        logger.info("Rule: {}".format(rule.__doc__))
        logger.info("kmin: {}, kmax: {}".format(n_clusts[0], n_clusts[1]))

        # skip if number of clusters is bigger than number of samples
        if n_clusts[1] >= n:
            logger.info("Kmax too large for dataset size. Skipping...")
            continue
        if n_clusts[0] <= 1:
            logger.info("Kmin too little. Skipping...")
            continue            

        for r in range(rounds): # for each round
            logger.info("Round: {}".format(r))

            results.round[res_idx] = r # round number
            results.n_samples[res_idx] = n # n_samples
            results.rule[res_idx] = rule.__doc__ # rule
            results.kmin[res_idx] = n_clusts[0] # kmin
            results.kmax[res_idx] = n_clusts[1] # kmax
            results.type_mat[res_idx] = mat_sparse # type of matrix
    
            logger.info("Generating ensemble...")

            generator = myKM.K_Means(cuda_mem="manual")
            
            t.tic()
            ensemble = part.generateEnsemble(data_sampled, generator, n_clusts,
                                             n_partitions, n_iters)
            t.tac()

            max_cluster_size = max([max(map(np.size,p)) for p in ensemble])
            
            results.t_ensemble[res_idx] = t.elapsed # ensemble time
            results.biggest_cluster[res_idx] = max_cluster_size # biggest_cluster

            logger.info("Sparse matrix: {}".format(mat_sparse))
            logger.info("Building matrix...")

            
            
            if not mat_sparse:
                myEst = eac.EAC(n, mat_sparse=False)
                t.tic()
                myEst.fit(ensemble, files=False, assoc_mode=assoc_mode,
                          prot_mode=prot_mode)
                t.tac()
                myEst._getAssocsDegree()
                # build time
                results.t_build[res_idx] = t.elapsed
                # number of associations
                results.n_assocs[res_idx] = myEst.getNNZAssocs()
                # stats number associations
                results.max_assoc[res_idx] = myEst.degree.max()
                results.std_assoc[res_idx] = myEst.degree.min()
                results.mean_assoc[res_idx] = myEst.degree.mean()
                results.std_assoc[res_idx] = myEst.degree.std()
            else:
                
                mymaxassocs = max_assocs_fn(ensemble) * 3
                myEst = mySparse.EAC_CSR(n_samples=n,
                                         max_assocs=mymaxassocs)
                t.tic()
                myEst._update_ensemble(ensemble, sort_mode="online")
                t.tac()
                # build time
                results.t_build[res_idx] = t.elapsed
                # number of associations
                results.n_assocs[res_idx] = myEst.nnz
                # stats number associations
                results.max_assoc[res_idx] = myEst.degree.max()
                results.std_assoc[res_idx] = myEst.degree.min()
                results.mean_assoc[res_idx] = myEst.degree.mean()
                results.std_assoc[res_idx] = myEst.degree.std()

            
            if mat_sparse: # don't do SL if sparse matrix -> NOT IMPLEMENTED
                results.to_csv(folder + "results_kmin.csv")
                res_idx += 1
                del generator, ensemble, myEst
                continue

            # logger.info("SL clustering...")

            # t.tic()
            # labels = myEst._lifetime_clustering()
            # t.tac()
            
            # logger.info("Scoring accuracy...")
            # accEst = accuracy.HungarianIndex(n)
            # accEst.score(gt_sampled, labels)
            
            # results.t_sl[res_idx] = t.elapsed # build time
            # results.accuracy[res_idx] = accEst.accuracy
            
            results.to_csv(folder + "results_kmin.csv")
            res_idx += 1

            del generator, ensemble, myEst#, accEst
            # end of inner most loop


    del data_sampled#, gt_sampled
    # end of dataset cycle