# needs:
#  - mixture
#  - numOracles
#  - numClusters
#  - qubitStringLen
#  - qGenerations
#  - 
# returns:
#  - qk_timings_cg
#  - qk_centroids (centroids of oracles from last generation)
#  - qk_assignment(assignment of oracles from last generation)
#  - fitnessEvolution
#  - qk_total
#  - qk_genTimes
#  - qk_assignedData

fitnessEvolution = np.zeros((qGenerations,numOracles+1))

qk_timings_cg = list()
start = datetime.now()

best = 0 #index of best oracle (starts at 0)

oras = list()
qk_centroids = [0]*numOracles
qk_estimator = [0]*numOracles
qk_assignment = [0]*numOracles

for i in range(0,numOracles):
    oras.append(oracle.Oracle())
    oras[i].initialization(numClusters*2,qubitStringLen)
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
        score = DaviesBouldin.DaviesBouldin(mixture,qk_centroids[i],qk_assignment[i])
        ora.score = score.eval()
    
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
    print '.', # simple "progress bar"
    start = datetime.now()
    
print 'Done!'

qk_genTimes=np.array(qk_timings_cg)
qk_total = np.sum(np.array(qk_timings_cg))
print "mean of generation:\t",np.mean(qk_genTimes),"\nvariance of generation:\t",np.var(qk_genTimes),"\ntotal time:\t",qk_total

# assign clusters to data
qk_assignedData = [None]*numClusters
for i,j in enumerate(qk_assignment[best]):
    if qk_assignedData[j] != None:
        qk_assignedData[j] = np.vstack((qk_assignedData[j],mixture[i]))
    else:
        qk_assignedData[j] = mixture[i]