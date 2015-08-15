
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

import MyML.helper.partition as part
import MyML.cluster.K_Means3 as myKM
import MyML.metrics.accuracy as myAcc
import MyML.utils.profiling as myProf
import MyML.EAC.eac_new as myEAC
import MyML.EAC.sparse as mySpEAC


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


## memory functions helper


full_mem = lambda n: n ** 2
full_condensed_mem = lambda n: np.arange(1,n-1).sum()
sp_cont_mem = lambda n, ma: n*ma*5 + 2*(n+1)

def sp_lin_mem(n, ma, n_s, n_e, val_s, val_e):
    l_rect = n * n_s * ma
    r_right = n    * (1.0 - n_e) * (val_e * ma)

    tri_base = n * (n_e - n_s)
    tri_height = ma * (val_s - val_e)
    tri_area = tri_base * tri_height / 2.0

    return l_rect + r_right + tri_area

def compute_mems(n, ma, n_s, n_e, val_s, val_e):
    full = full_mem(n)
    full_cond = full_condensed_mem(n)
    sp = sp_cont_mem(n, ma)
    sp_lin = sp_lin_mem(n, ma, n_s, n_e, val_s, val_e)
    return full, full_cond, sp, sp_lin


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

mem_full_max = 25 * 2**30 # max mem full mat can take in bytes, 2**30 = 1GB

# number of samples
cardinality = [1e2,2.5e2,5e2,7.5e2,
               1e3,2.5e3,5e3,7.5e3,
               1e4,2.5e4,5e4,7.5e4,
               1e5,2.5e5,5e5,7.5e5,
               1e6,2.5e6]
cardinality = map(int,cardinality)

total_n = data.shape[0]
div = map(lambda n: total_n / n, cardinality)

# prepare results datastructure
res_cols = ['n_samples',
            'rule',
            'kmin', 'kmax',
            't_ensemble', 't_build', 't_sl', 't_accuracy',
            'biggest_cluster',
            'type_mat',
            'n_assocs', 'n_max_degree',
            'min_degree', 'max_degree', 'mean_degree', 'std_degree',
            'accuracy', 'sl_clusts',
            'round']

type_mats = ["full",
             "full condensed",
             "sparse complete",
             "sparse condensed const",
             "sparse condensed linear"]
rounds = 5
res_lines = rounds * len(cardinality) * len(rules) * len(type_mats)

results = pd.DataFrame(index=range(res_lines), columns=res_cols)


t = myProf.Timer() # timer

# ensemble properties
n_partitions = 100
n_iters = 3

# EAC properties
sparse_max_assocs_factor = 3

# ## run

logger.info("Starting experiment...")

# In[198]:

res_idx = 0
for d in div: # for each size of dataset
   
    # sample data
    data_sampled = np.ascontiguousarray(data[::d])
    gt_sampled = np.ascontiguousarray(gt[::d])
    n = data_sampled.shape[0]

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


        ## generate ensemble
        logger.info("Generating ensemble...")

        generator = myKM.K_Means(cuda_mem="manual")
        
        t.tic()
        ensemble = part.generateEnsemble(data_sampled, generator, n_clusts,
                                         n_partitions, n_iters)
        t.tac()

        t_ensemble = t.elapsed

        max_cluster_size = myEAC.biggest_cluster_size(ensemble)


        # check memory usage for different matrix schemes

        # compute memory usage for each type of matrix
        # linear properties for condensed sparse matrix
        n_s = 0.05
        n_e = 1.0
        val_s = 1.0
        val_e = 0.05

        ma = max_cluster_size * sparse_max_assocs_factor

        mems = compute_mems(n, ma, n_s, n_e, val_s, val_e)
        mems = map(lambda x: True if x < mem_full_max else False, mems)

        # if all matrices exceed max memory don't build matrix
        # and don't increment pointer of results, i.e. overwrite this iter.
        if sum(mems) == 0:
            continue

        f_mat = mems[0] # full matrix
        fc_mat = mems[1] # full condensed matrix
        sp_const = mems[2] # sparse constant matrix
        sp_lin = mems[3] # sparse linear matrix    

        for tm in xrange(len(type_mats)):
            logger.info("Type of mat: {}".format(type_mats[tm]))

            for r in range(rounds): # for each round

                logger.info("Round: {}".format(r))

                logger.info("Building matrix...")

                if tm == 0: # full 
                    if not f_mat:
                        logger.info("not enough memory")
                        break
                    eacEst = myEAC.EAC(n_samples=n, sparse=False, condensed=False)
                    t.tic()
                    eacEst.buildMatrix(ensemble)
                    t.tac()

                    eacEst.coassoc.getDegree()
                    degree = eacEst.coassoc.degree
                    nnz = eacEst.coassoc.nnz

                    n_max_degree = -1

                elif tm == 1: # full condensed
                    if not fc_mat:
                        logger.info("not enough memory")
                        break
                    eacEst = myEAC.EAC(n_samples=n, sparse=False, condensed=True)
                    t.tic()
                    eacEst.buildMatrix(ensemble)
                    t.tac()

                    eacEst.coassoc.getDegree()
                    degree = eacEst.coassoc.degree
                    nnz = eacEst.coassoc.nnz

                    n_max_degree = -1

                elif tm == 2: # sparse complete
                    if not sp_const:
                        logger.info("not enough memory")
                        break
                    eacEst = myEAC.EAC(n_samples=n, sparse=True, condensed=False,
                                       sparse_keep_degree=True)
                    eacEst.sp_max_assocs_mode="constant"
                    t.tic()                    
                    eacEst.buildMatrix(ensemble)
                    t.tac()

                    degree = eacEst.coassoc.degree[:-1]
                    nnz = eacEst.coassoc.nnz

                    n_max_degree = (degree == ma).sum()

                elif tm == 3: # sparse condensed const
                    if not sp_const:
                        logger.info("not enough memory")
                        break
                    eacEst = myEAC.EAC(n_samples=n, sparse=True, condensed=True,
                                       sparse_keep_degree=True)
                    eacEst.sp_max_assocs_mode="constant"
                    t.tic()                    
                    eacEst.buildMatrix(ensemble)
                    t.tac()

                    degree = eacEst.coassoc.degree[:-1]
                    nnz = eacEst.coassoc.nnz

                    n_max_degree = (degree == ma).sum()

                elif tm == 4: # sparse condensed linear
                    if not sp_lin:
                        logger.info("not enough memory")
                        break
                    eacEst = myEAC.EAC(n_samples=n, sparse=True, condensed=True,
                                       sparse_keep_degree=True)
                    eacEst.sp_max_assocs_mode="linear"
                    t.tic()                    
                    eacEst.buildMatrix(ensemble)
                    t.tac()

                    degree = eacEst.coassoc.degree[:-1]
                    nnz = eacEst.coassoc.nnz

                    indptr = mySpEAC.indptr_linear(n,
                                                   eacEst.sp_max_assocs,
                                                    n_s, n_e, val_s, val_e)
                    max_degree = indptr[1:] - indptr[:-1]
                    n_max_degree = (degree == max_degree).sum()

                else:
                    raise NotImplementedError("mat type {} not implemented".format(type_mats[tm]))

                logger.info("Build time: {}".format(t.elapsed))

                results.round[res_idx] = r # round number
                results.n_samples[res_idx] = n # n_samples
                results.rule[res_idx] = rule.__doc__ # rule
                results.kmin[res_idx] = n_clusts[0] # kmin
                results.kmax[res_idx] = n_clusts[1] # kmax
                results.t_build[res_idx] = t.elapsed
                results.type_mat[res_idx] = type_mats[tm] # type of matrix    

                results.t_ensemble[res_idx] = t.elapsed # ensemble time
                results.biggest_cluster[res_idx] = max_cluster_size # biggest_cluster

                # number of associations
                results.n_assocs[res_idx] = nnz

                # stats number associations
                results.max_degree[res_idx] = degree.max()
                results.min_degree[res_idx] = degree.min()
                results.mean_degree[res_idx] = degree.mean()
                results.std_degree[res_idx] = degree.std()
                results.n_max_degree[res_idx] = n_max_degree
                
                # if mat_sparse: # don't do SL if sparse matrix -> NOT IMPLEMENTED
                #     results.to_csv(folder + "results_kmin.csv")
                #     res_idx += 1
                #     del generator, ensemble, myEst
                #     continue

                logger.info("SL clustering...")

                t.tic()
                labels = eacEst.finalClustering(n_clusters=0)
                t.tac()
                logger.info("Clustering time: {}".format(t.elapsed))

                results.t_sl[res_idx] = t.elapsed # build time
                results.sl_clusts[res_idx] = eacEst.n_fclusts

                t.tic()
                # logger.info("Scoring accuracy (consistency)...")
                # accEst = myAcc.ConsistencyIndex(n)
                # accEst.score(gt_sampled, labels)

                logger.info("Scoring accuracy (Hungarian)...")
                accEst = myAcc.HungarianIndex(n)
                accEst.score(gt_sampled, labels)
                
                t.tac()

                logger.info("Accuracy time: {}".format(t.elapsed))

                results.t_accuracy[res_idx] = t.elapsed # accuracy time
                results.accuracy[res_idx] = accEst.accuracy
                
                results.to_csv(folder + "results_kmin.csv")
                res_idx += 1

                del eacEst, accEst
                # end of inner most loop


    del data_sampled, gt_sampled
    # end of dataset cycle