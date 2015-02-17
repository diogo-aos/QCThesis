import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

import oracle
import qubitLib
import DaviesBouldin

import QK_Means
import K_Means

import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="where to read data from",type=str)
parser.add_argument("-c", "--clusters", help="number of clusters",type=int)
parser.add_argument("-o", "--oracles", help="number of oracles",type=int)
parser.add_argument("-s", "--stringLen", help="length of qubit strings",type=int)
parser.add_argument("-g", "--generations", help="number of generations",type=int)
parser.add_argument("-r", "--rounds", help="number of rounds",type=int)
parser.add_argument("-f", "--factor", help="iteration factor for K-Means",type=float)
args = parser.parse_args()

#########################################################
# file names
filenames=dict()
filenames['data']=args.data
filenames['qk']=args.data+'_'+'qk'
filenames['k']=args.data+'_'+'k'


# Quantum parameters
numClusters=args.clusters
numOracles=args.oracles
qubitStringLen=args.stringLen
qGenerations=args.generations
numRounds=args.rounds

# Normal parameters
initsPercentage=args.factor # multiplier factor for numOracles * qGenerations for the number of inits

# load data from file
pkl_file = open(filenames['data'], 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

mixture=data['data']
dim=data['dim']

del data

#########################################################

## Quantum K-Means

#########################################################

print 'Initiating QK-Means...'

qk_results=list()
for i in range(numRounds):

    start = datetime.now()

    qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg=QK_Means.qk_means(mixture,numOracles,numClusters,qubitStringLen,qGenerations,dim)
    qk_results.append([qk_centroids,qk_assignment,fitnessEvolution,qk_timings_cg])

    round=(datetime.now() - start).total_seconds()
    print float(i+1)*100/numRounds,'%\t','round ', i,':',round,'s  -  estimated:',(float(numRounds-1)-i)*round,'s / ',(float(numRounds-1)-i)*round/60,'m'



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
qk_rounds['assignedData']=[None]*numRounds #assigned data for the best solution in each round

for r in range(numRounds):
    # only assign for best solution
    best=int(qk_rounds['fitness'][r][-1,-1])
    qk_assignment=qk_rounds['assignment'][r]

    # store only best centroids
    qk_rounds['centroids'][r]=qk_rounds['centroids'][r][best]

    qk_assignedData = [None]*numClusters
    for i,j in enumerate(qk_assignment[best]):
        if qk_assignedData[j] != None:
            qk_assignedData[j] = np.vstack((qk_assignedData[j],mixture[i]))
        else:
            qk_assignedData[j] = mixture[i]
    
    qk_rounds['assignedData'][r]=qk_assignedData


# convert computation times
qk_rounds['total time'] = list()

for i in range(numRounds):
    qk_times=qk_rounds['times'][i]
    qk_total = np.sum(np.array(qk_times))
    qk_rounds['total time'].append(qk_total)


# compute population's fitness evolution (best solution, mean, variance) per generation
qk_rounds['best evolution'] = [None]*numRounds
qk_rounds['pop variance'] = [None]*numRounds
qk_rounds['pop mean'] = [None]*numRounds

for i in range(numRounds):
    bestSeq=[]
    bestSeqIndex=[]
    fitnessEvolution=qk_rounds['fitness'][i]
    for j in range(0,fitnessEvolution.shape[0]):
        bestSeq.append(fitnessEvolution[j,fitnessEvolution[j,-1]])
    bestSeq=np.array(bestSeq)
    
    genVar=np.zeros(qGenerations)
    genMean=np.zeros(qGenerations)
    
    for j,ar in enumerate(fitnessEvolution):
        genVar[j]=np.var(ar[:-1])
        genMean[j]=np.mean(ar[:-1])
        
    qk_rounds['best evolution'][i]=bestSeq
    qk_rounds['pop variance'][i]=genVar
    qk_rounds['pop mean'][i]=genMean


qk_rounds['fitness best']=[None]*numRounds
qk_rounds['fitness worst']=[None]*numRounds
qk_rounds['fitness mean']=[None]*numRounds
qk_rounds['fitness variance']=[None]*numRounds

for i in range(numRounds):
	qk_rounds['fitness best'][i]=qk_rounds['best evolution'][i][-1]
	qk_rounds['fitness worst'][i]=np.min(qk_rounds['fitness'][i])
	qk_rounds['fitness mean']=qk_rounds['pop mean'][i][-1]
	qk_rounds['fitness variance']=qk_rounds['pop variance'][i][-1]

del qk_results
del qk_rounds['fitness']
del qk_rounds['times']
del qk_rounds['assignment']


#print "keys stored:"
#print qk_rounds.keys()

# save to file
output = open(filenames['qk']+'.pkl', 'wb')
pickle.dump(qk_rounds, output)
output.close()
print 'Data saved to '+filenames['qk']+'.pkl'

#########################################################

print 'Initiating K-Means...'

numInits=np.int(qGenerations*numOracles*initsPercentage)

print "Each round will have ",numInits," K-Means initializations."

k_results=list()

for i in range(numRounds):
    start=datetime.now()

    k_centroids,k_assignment,k_timings=K_Means.k_means(mixture,numClusters,numInits)
    k_results.append([k_centroids,k_assignment,k_timings])

    round=(datetime.now() - start).total_seconds()
    print float(i+1)*100/numRounds,'%\t','round ', i,':',round,'s  -  estimated:',(float(numRounds-1)-i)*round,'s / ',(float(numRounds-1)-i)*round/60,'m'

print 'Preparing K-Means data structures...'

k_rounds=dict()
k_rounds['centroids']=list()
k_rounds['assignment']=list()
k_rounds['times']=list()

for i in range(numRounds):
    k_rounds['centroids'].append(k_results[i][0])
    k_rounds['assignment'].append(k_results[i][1])
    k_rounds['times'].append(k_results[i][2])
    

# compute computation times
k_rounds['total time'] = list()

for i in range(numRounds):
    k_times=k_rounds['times'][i]
    k_total = np.sum(np.array(k_times))
    k_rounds['total time'].append(k_total)

#compute fitnesses
k_rounds['fitness']=[np.inf]*numRounds
k_rounds['fitness best']=[np.inf]*numRounds
k_rounds['fitness worst']=[0]*numRounds
k_rounds['fitness best index']=[None]*numRounds
k_rounds['best centroids']=[None]*numRounds

print 'Computing fitness score...'
for i in range(numRounds):
    start=datetime.now()
    k_rounds['fitness'][i]=[None]*numInits
    for j in range(numInits):
        k_score=DaviesBouldin.DaviesBouldin(mixture,k_rounds['centroids'][i][j],k_rounds['assignment'][i][j])
        k_rounds['fitness'][i][j]=k_score.eval()

        # store best score & index
        if k_rounds['fitness'][i][j]<k_rounds['fitness best'][i]:
        	k_rounds['fitness best'][i]=k_rounds['fitness'][i][j]
        	k_rounds['fitness best index'][i]=j
        # store worst score
        if  k_rounds['fitness'][i][j]>k_rounds['fitness worst'][i]:
        	k_rounds['fitness worst'][i]= k_rounds['fitness'][i][j]

    # store fitness' mean and variance of current round
    k_rounds['fitness mean']=np.mean(k_rounds['fitness'][i])
    k_rounds['fitness variance']=np.var(k_rounds['fitness'][i])

    # store centroids of fittest solution
    k_rounds['best centroids'][i]=k_rounds['centroids'][i][k_rounds['fitness best index'][i]]

    round=(datetime.now() - start).total_seconds()
    print float(i+1)*100/numRounds,'%\t','round ', i,':',round,'s  -  estimated:',(float(numRounds-1)-i)*round,'s / ',(float(numRounds-1)-i)*round/60,'m'


k_rounds['centroids']=k_rounds['best centroids']



# assign data to clusters
k_rounds['assignedData']=[None]*numRounds #assigned data for the best solution in each round

for r in range(numRounds):
    # assign clusters to data
    k_assignment=k_rounds['assignment'][r][k_rounds['fitness best index'][r]]
    
    k_assignedData = [None]*numClusters
    for i,j in enumerate(k_assignment):
        if k_assignedData[j] != None:
            k_assignedData[j] = np.vstack((k_assignedData[j],mixture[i]))
        else:
            k_assignedData[j] = mixture[i]
    
    k_rounds['assignedData'][r]=k_assignedData

del k_results
del k_rounds['assignment']
del k_rounds['fitness best index']
del k_rounds['best centroids']
del k_rounds['fitness']
del k_rounds['times']

#print "keys stored:"
#print k_rounds.keys()

# save to file
output = open(filenames['k']+'.pkl', 'wb')
pickle.dump(k_rounds, output)
output.close()
print 'Data saved to '+filenames['k']+'.pkl'