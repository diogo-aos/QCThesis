import numpy as np
from sklearn import preprocessing

def pcaFun(x, whiten=False):
	avg=np.mean(x,axis=0)

	# center data
	cX=x-avg

	# compute covariance matrix
	C=cX.T.dot(cX)

	# compute eig
	eigVals,eigVect=np.linalg.eig(C)

	# get decresing order of eigVals (index)
	eigValOrder=eigVals.argsort()[::-1]

	#sort eigenthings
	sortedEigVect=np.zeros(eigVect.shape)
	sortedEigVal=np.zeros(eigVals.shape)

	for i,j in enumerate(eigValOrder):
		sortedEigVect[:,i]=eigVect[:,j]
		sortedEigVal[i]=eigVals[j]

	# project data
	projX=cX.dot(sortedEigVect)

	if whiten is True:
		e = 0.00005
		projX = projX / ((sortedEigVal + e) ** 0.5)

	return projX



# function graddesc(xyData,q,[steps])
# purpose: performing quantum clustering in and moving the 
#          data points down the potential gradient
# input: xyData - the data vectors
#        q=a parameter for the parsen window variance (q=1/(2*sigma^2))
#		 sigma=parameter for the parsen window variance (choose q or sigma)
#        steps=number of gradient descent steps (default=50)
#		 eta=gradient descent step size
# output: D=location of data o=point after GD 

def graddesc(xyData,**kwargs):

	argKeys = kwargs.keys()

	if 'steps' in argKeys:
		steps = kwargs['steps']
	else:
		steps = 50

	if 'q' in argKeys:
		q = kwargs['q']
	elif 'sigma' in argKeys:
		sigma = kwargs['sigma']
		q = 1 / (2 * pow(sigma,2))
	else:
		sigma=0.1
		q = 1 / (2 * pow(sigma,2))

	if 'r' in argKeys:
		D = kwargs['r']
	else:
		D = xyData

	if 'eta' in argKeys:
		eta = kwargs['eta']
	else:
		eta = 0.1

	if 'all_square' in argKeys and kwargs['all_square'] is not False:
		if xyData.shape[1]>2:
			raise Exception('all_square should not be used in data > 2 dims')
		points=kwargs['all_square']
		totalPoints=pow(kwargs['all_square'],2)
		a=np.linspace(-1,1,points)
		D=[(x,y) for x in a for y in a]
		D=np.array(D)
	else:
		D=xyData

	if 'return_eta' in argKeys:
		return_eta=kwargs['return_eta']
	else:
		return_eta=False

	if 'timelapse' in argKeys:
		timelapse=kwargs['timelapse']
		if timelapse:
			tD=list()
			timelapse_count=0
			if 'timelapse_list' in argKeys:
				timelapse_list=kwargs['timelapse_list']
			elif 'timelapse_percent' in argKeys:
				timelapse_percent=kwargs['timelapse_percent']
				timelapse_list=range(steps)[::int(steps*timelapse_percent)]
			else:
				timelapse_percent=0.25
				timelapse_list=range(steps)[::int(steps*timelapse_percent)]
	else:
		timelapse=False

	# add more states to timelapse list
	if timelapse:
		if timelapse_count in timelapse_list:
			tD.append(D)
		timelapse_count += 1	

	# first run
	V,P,E,dV = qc(xyData,q=q,r=D)

	for j in range(4):
		for i in range(steps/4):
			# normalize potential gradient
			dV = preprocessing.normalize(dV)
			
			# gradient descent
			D = D - eta*dV

			# add more states to timelapse list
			if timelapse:
				if timelapse_count in timelapse_list:
					tD.append(D)
				timelapse_count += 1			

			# perform Quantum Clustering
			V,P,E,dV = qc(xyData,q=q,r=D)
		eta*=0.5

	if timelapse:
		tD.append(D)
		D=tD

	if return_eta:
		return D,V,E,eta
	return D,V,E
	

# function qc
# purpose: performing quantum clustering in n dimensions
# input:
#       ri - a vector of points in n dimensions
#       q - the factor q which determines the clustering width
#       r - the vector of points to calculate the potential for. equals ri if not specified
# output:
#       V - the potential
#       P - the wave function
#       E - the energy
#       dV - the gradient of V
# example: [V,P,E,dV] = qc ([1,1;1,3;3,3],5,[0.5,1,1.5]);
# see also: qc2d

def qc(ri,**kwargs):
	argKeys=kwargs.keys()

	if 'q' in argKeys:
		q=kwargs['q']
	elif 'sigma' in argKeys:
		sigma=kwargs['sigma']
		q = 1 / (2 * pow(sigma,2))
	else:
		sigma=0.1
		q = 1 / (2 * pow(sigma,2))

	if 'r' in argKeys:
		r=kwargs['r']
	else:
		r=ri

	pointsNum,dims = ri.shape
	calculatedNum = r.shape[0]

	# prepare the potential
	V=np.zeros(calculatedNum)
	dP2=np.zeros(calculatedNum)

	# prepare P
	P=np.zeros(calculatedNum)
	singledV1=np.zeros((calculatedNum,dims))
	singledV2=np.zeros((calculatedNum,dims))

	dV1=np.zeros((calculatedNum,dims))
	dV2=np.zeros((calculatedNum,dims))
	dV=np.zeros((calculatedNum,dims))

	# prevent division by zero
	# calculate V
	# run over all the points and calculate for each the P and dP2

	for point in range(calculatedNum):

		# compute ||x-xi||^2
		# axis=1 will sum rows instead of columns
		D2 = np.sum(pow(r[point]-ri,2),axis=1)

		# compute gaussian
		singlePoint = np.exp(-q*D2)

		# compute Laplacian of gaussian = ||x-xi||^2 * exp(...)
		singleLaplace = D2 * singlePoint

		#compute gradient components
		aux = r[point] - ri
		for d in range(dims):
			singledV1[:,d] = aux[:,d] * singleLaplace
			singledV2[:,d] = aux[:,d] * singlePoint

		P[point] = np.sum(singlePoint)
		dP2[point] = np.sum(singleLaplace)
		dV1[point] = np.sum(singledV1,axis=0)
		dV2[point] = np.sum(singledV2,axis=0)

	# if there are points with 0 probability, 
	# assigned them the lowest probability of any point
	P=np.where(P==0,np.min(np.extract((P!=0),P)),P)

	# compute ground state energy
	V = -dims/2 + q*dP2 / P
	E = -min(V)

	# compute potential on points
	V += E

	# compute gradient of V
	for d in range(dims):
		dV[:,d] = -q * dV1[:,d] + (V-E+(dims+2)/2) * dV2[:,d]

	return V,P,E,dV

# clust=fineCluster(xyData,minD) cluster xyData points when closer than minD
# output: clust=vector the cluter index that is asigned to each data point
#        (it's cluster serial #)
def fineCluster(xyData,minD):
	
	n = xyData.shape[0]
	clust = np.zeros(n)
	i=0
	clustInd=1
	while np.min(clust)==0:
		x=xyData[i]

		# euclidean distance form 1 point to others
		D = np.sum(pow(x-xyData,2),axis=1)
		D = pow(D,0.5)

		clust = np.where(D<minD,clustInd,clust)

		#index of first unclestered datapoint
		i=np.argmin(clust)

		clustInd += 1

	return clust

def fineCluster2(xyData,pV,minD):
	
	n = xyData.shape[0]
	clust = np.zeros(n)

	# index of points sorted by potential
	sortedUnclust=pV.argsort()

	# index of unclestered point with lowest potential
	i=sortedUnclust[0]

	# fist cluster index is 1
	clustInd=1

	while np.min(clust)==0:
		x=xyData[i]

		# euclidean distance form 1 point to others
		D = np.sum(pow(x-xyData,2),axis=1)
		D = pow(D,0.5)

		clust = np.where(D<minD,clustInd,clust)
		
		# index of non clustered points
		# unclust=[x for x in clust if x == 0]
		clusted= clust.nonzero()[0]

		# sorted index of non clustered points
		sortedUnclust=[x for x in sortedUnclust if x not in clusted]

		if len(sortedUnclust) == 0:
			break

		#index of unclestered point with lowest potential
		i=sortedUnclust[0]

		clustInd += 1

	return clust


