import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

import oracle
import qubitLib
import DaviesBouldin


# Receives:
#  - mixture
#  - numOracles
#  - numClusters
#  - qubitStringLen
#  - qGenerations
#  - 
# Returns:
#  - qk_timings_cg
#  - qk_centroids (centroids of oracles from last generation)
#  - qk_assignment(assignment of oracles from last generation)
#  - fitnessEvolution
def qk_means(mixture,numOracles,numClusters,qubitStringLen,qGenerations,dim,timeDB=False):
    fitnessEvolution = np.zeros((qGenerations,numOracles+1)) #+1 for the column that indicates the best score in each gen

    if timeDB:
        db_timings = list() #timing list for Davies-Bouldin index computation
    qk_timings_cg = list()
    start = datetime.now()
    
    best = 0 #index of best oracle (starts at 0)
    
    oras = list()
    qk_centroids = [0]*numOracles
    qk_estimator = [0]*numOracles
    qk_assignment = [0]*numOracles
    
    for i in range(0,numOracles):
        oras.append(oracle.Oracle())
        oras[i].initialization(numClusters*dim,qubitStringLen)
        oras[i].collapse()
    
    qk_timings_cg.append((datetime.now() - start).total_seconds())
    start = datetime.now()
    
    for qGen_ in range(0,qGenerations):
        ## Clustering step
        for i,ora in enumerate(oras):
            if qGen_ != 0 and i == best: # current best shouldn't be modified
                continue 
            qk_centroids[i] = np.vstack(np.hsplit(ora.getIntArrays(),numClusters))
            qk_estimator[i] = KMeans(n_clusters=numClusters,init=qk_centroids[i],n_init=1)
            qk_assignment[i] = qk_estimator[i].fit_predict(mixture)
            qk_centroids[i] = qk_estimator[i].cluster_centers_
            ora.setIntArrays(np.concatenate(qk_centroids[i]))
        
        ## Compute fitness
            if timeDB:
                db_start = datetime.now()
            score = DaviesBouldin.DaviesBouldin(mixture,qk_centroids[i],qk_assignment[i])
            ora.score = score.eval()

            if timeDB:
                db_timings.append((datetime.now() - db_start).total_seconds())

        
        ## Store best from this generation
        for i in range(1,numOracles):
            if oras[i].score < oras[best].score:
                best = i
                
        ## Quantum Rotation Gate 
        for i in range(0,numOracles):
            if i == best:
                continue
            
            oras[i].QuantumGateStep(oras[best])
            
        ## Collapse qubits
            oras[i].collapse()
            
        qk_timings_cg.append((datetime.now() - start).total_seconds())
    
        for i in range(0,numOracles):
            fitnessEvolution[qGen_,i]=oras[i].score
            fitnessEvolution[qGen_,-1]=best

        start = datetime.now()

    if timeDB:
        return qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg,db_timings

    return qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg