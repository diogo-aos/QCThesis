import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

import oracle
import qubitLib
import DaviesBouldin

import QK-Means
import K-Means

import pickle
#########################################################
# file names
filenames=dict()
filenames['data']='data'
filenames['qk']='qk_rounds'
filenames['k']='k_rounds'


# gaussian parameters
numPoints=200
numGaussians=6

# Quantum parameters
numClusters=15
numOracles=5
qubitStringLen=5
qGenerations=100
numRounds=5

# Normal parameters
initsPercentage=1.15 # multiplier factor for numOracles * qGenerations for the number of inits


#########################################################

# generate gaussians

pointsPerGaussian=numPoints/numGaussians
gMix=list()
gMix.append(np.random.normal((-5,3),[0.25,2],(pointsPerGaussian,2)))
gMix.append(np.random.normal((2,-2),[0.25,0.25],(pointsPerGaussian,2)))
gMix.append(np.random.normal((2,10),[0.75,0.25],(pointsPerGaussian,2)))
gMix.append(np.random.normal((-3,2),[0.3,0.3],(pointsPerGaussian,2)))
gMix.append(np.random.normal((3,7),[0.3,0.3],(pointsPerGaussian,2)))
gMix.append(np.random.normal((-2,7),[0.3,1],(pointsPerGaussian,2)))


mixture=np.concatenate(tuple(gMix))

# save to file
output = open(filenames['data']+'.pkl', 'wb')
pickle.dump(mixture, output)
output.close()
print 'Data saved to '+filenames['data']+'.pkl'

#########################################################

## Quantum K-Means

#########################################################

print 'Initiating QK-Means...'
qk_results=list()
for i in range(numRounds):
    print 'round ',i
    qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg=qk_means(mixture,numOracles,numClusters,qubitStringLen,qGenerations)
    qk_results.append([qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg])

#########################################################
print 'Preparing data structures...'

qk_rounds=dict()
qk_rounds['centroids']=list()
qk_rounds['assignment']=list()
qk_rounds['fitness']=list()
qk_rounds['times']=list()

for i in range(numRounds):
    qk_rounds['centroids'].append(qk_results[i][0])
    qk_rounds['assignment'].append(qk_results[i][1])
    qk_rounds['fitness'].append(qk_results[i][2])
    qk_rounds['times'].append(qk_results[i][3])
    
# assign data to clusters
qk_rounds['assignedData']=list() #assigned data for the best solution in each round

for i in range(numRounds):
    # assign clusters to data
    best=int(qk_rounds['fitness'][i][-1,-1])
    qk_assignment=qk_rounds['assignment'][i]
    
    qk_assignedData = [None]*numClusters
    for i,j in enumerate(qk_assignment[best]):
        if qk_assignedData[j] != None:
            qk_assignedData[j] = np.vstack((qk_assignedData[j],mixture[i]))
        else:
            qk_assignedData[j] = mixture[i]
    
    qk_rounds['assignedData'].append(qk_assignedData)

# convert computation times
qk_rounds['total time'] = list()

for i in range(numRounds):
    qk_times=qk_rounds['times'][i]
    qk_total = np.sum(np.array(qk_times))
    qk_rounds['total time'].append(qk_total)

# convert population fitness data
qk_rounds['best evolution'] = list()
qk_rounds['pop variance'] = list()
qk_rounds['pop mean'] = list()

for i in range(numRounds):
    bestSeq=[]
    bestSeqIndex=[]
    fitnessEvolution=qk_rounds['fitness'][i]
    for i in range(0,fitnessEvolution.shape[0]):
        bestSeq.append(fitnessEvolution[i,fitnessEvolution[i,-1]])
    bestSeq=np.array(bestSeq)
    
    genVar=np.zeros(qGenerations)
    genMean=np.zeros(qGenerations)
    
    for i,ar in enumerate(fitnessEvolution):
        genVar[i]=np.var(ar[:-1])
        genMean[i]=np.mean(ar[:-1])
        
    qk_rounds['best evolution'].append(bestSeq)
    qk_rounds['pop variance'].append(genVar)
    qk_rounds['pop mean'].append(genMean)


# save to file
output = open(filenames['qk']+'.pkl', 'wb')
pickle.dump(qk_rounds, output)
output.close()
print 'Data saved to '+filenames['qk']+'.pkl'

#########################################################

print 'Initiating K-Means...'

numInits=np.int(qGenerations*numOracles*initsPercentage)

k_results=list()

for i in range(numRounds):
    print 'round ',i
    k_centroids,k_assignment,k_timings=k_means(mixture,numClusters,numInits)
    k_results.append([k_centroids,k_assignment,k_timings])

print 'Preparing K-Means data structures...'

k_rounds=dict()
k_rounds['centroids']=list()
k_rounds['assignment']=list()
k_rounds['times']=list()

for i in range(numRounds):
    k_rounds['centroids'].append(k_results[i][0])
    k_rounds['assignment'].append(k_results[i][1])
    k_rounds['times'].append(k_results[i][2])
    
# assign data to clusters
k_rounds['assignedData']=list() #assigned data for the best solution in each round

for i in range(numRounds):
    # assign clusters to data
    best=int(qk_rounds['fitness'][i][-1,-1])
    k_assignment=k_rounds['assignment'][i]
    
    k_assignedData = [None]*numClusters
    for i,j in enumerate(k_assignment[best]):
        if k_assignedData[j] != None:
            k_assignedData[j] = np.vstack((k_assignedData[j],mixture[i]))
        else:
            k_assignedData[j] = mixture[i]
    
    k_rounds['assignedData'].append(k_assignedData)
    

# compute computation times
k_rounds['total time'] = list()

for i in range(numRounds):
    k_times=k_rounds['times'][i]
    k_total = np.sum(np.array(k_times))
    k_rounds['total time'].append(k_total)

k_time_mean = np.mean(np.array(k_rounds['total time']))
k_time_var = np.var(np.array(k_rounds['total time']))

k_rounds['fastest round']=np.argmin(np.array(k_rounds['total time']))
k_rounds['slowest round']=np.argmax(np.array(k_rounds['total time']))

k_time_best = k_rounds['total time'][k_rounds['fastest round']]
k_time_worst = k_rounds['total time'][k_rounds['slowest round']]

#compute fitnesses
k_rounds['fitness']=list()
start=datetime.now()
k_bestScore=np.inf
for i in range(numRounds):
    print 'round ',i
    init_scores=list()
    for j in range(numInits):
        k_score=DaviesBouldin.DaviesBouldin(mixture,k_rounds['centroids'][i][j],k_rounds['assignment'][i][j])
        init_scores.append(k_score.eval())
    k_rounds['fitness'].append(np.array(init_scores))
    m=np.min(init_scores)
    if m < k_bestScore:
        k_bestScore=m
        k_rounds['best fit round']=i

# save to file
output = open(filenames['k']+'.pkl', 'wb')
pickle.dump(k_rounds, output)
output.close()
print 'Data saved to '+filenames['k']+'.pkl'