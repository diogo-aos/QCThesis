

    cd /home/chiroptera/workspace/QCThesis/Horn

    /home/chiroptera/workspace/QCThesis/Horn



    %pylab inline

    Populating the interactive namespace from numpy and matplotlib



    import seaborn as sns
    import sklearn
    from sklearn import preprocessing,decomposition,datasets
    import HornAlg


    reload(HornAlg)




    <module 'HornAlg' from 'HornAlg.py'>




    # These are the "Tableau 20" colors as RGB.  
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
    for i in range(len(tableau20)):  
        r, g, b = tableau20[i]  
        tableau20[i] = (r / 255., g / 255., b / 255.)

# Quantum Clustering with Schrödinger's equation

## Background

This method starts off by creating a Parzen-window density estimation of the
input data by associating a Gaussian with each point, such that

$$ \psi (\mathbf{x}) = \sum ^N _{i=1} e^{- \frac{\left \|
\mathbf{x}-\mathbf{x}_i \right \| ^2}{2 \sigma ^2}} $$

where $N$ is the total number of points in the dataset, $\sigma$ is the variance
and $\psi$ is the probability density estimation. $\psi$ is chosen to be the
wave function in Schrödinger's equation. The details of why this is are better
described in [1-4]. Schrödinger's equation is solved in order of the potential
function $V(x)$, whose minima will be the centers of the clusters of our data:

$$
V(\mathbf{x}) = E + \frac {\frac{\sigma^2}{2}\nabla^2 \psi }{\psi}
= E - \frac{d}{2} + \frac {1}{2 \sigma^2 \psi} \sum ^N _{i=1} \left \|
\mathbf{x}-\mathbf{x}_i \right \| ^2 e^{- \frac{\left \| \mathbf{x}-\mathbf{x}_i
\right \| ^2}{2 \sigma ^2}}
$$

And since the energy should be chosen such that $\psi$ is the groundstate (i.e.
eigenstate corresponding to minimum eigenvalue) of the Hamiltonian operator
associated with Schrödinger's equation (not represented above), the following is
true

$$
E = - min \frac {\frac{\sigma^2}{2}\nabla^2 \psi }{\psi}
$$

With all of this, $V(x)$ can be computed. However, it's very computationally
intensive to compute V(x) to the whole space, so we only compute the value of
this function close to the datapoints. This should not be problematic since
clusters' centers are generally close to the datapoints themselves. Even so, the
minima may not lie on the datapoints themselves, so what we do is compute the
potential at all datapoints and then apply the gradient descent method to move
them to regions in space with lower potential.

There is another method to evolve the system other then by gradient descent
which is explained in [4] and complements this on the Dynamic Quantum Clustering
algorithm.

The code for this algorithm is available in Matlab in one of the authur's
webpage ([David Horn](http://horn.tau.ac.il/QC.htm)). That code has been ported
to Python and the version used in this notebook can be found [here](https://gith
ub.com/Chiroptera/QCThesis/blob/95020790d605cff1791810893439d419e16962d6/Horn/Ho
rnAlg.py).

## References

[1] D. Horn and A. Gottlieb, “The Method of Quantum Clustering.,” NIPS, no. 1,
2001.

[2] D. Horn, T. Aviv, A. Gottlieb, H. HaSharon, I. Axel, and R. Gan, “Method and
Apparatus for Quantum Clustring,” 2010.

[3] D. Horn and A. Gottlieb, “Algorithm for Data Clustering in Pattern
Recognition Problems Based on Quantum Mechanics,” Phys. Rev. Lett., vol. 88, no.
1, pp. 1–4, 2001.

[4] M. Weinstein and D. Horn, “Dynamic quantum clustering: a method for visual
exploration of structures in data,” pp. 1–15.


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
    		D = np.sum((x-xyData)**2,axis=1)
    		D = D**0.5
    
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

# Iris

The iris dataset ([available at the UCI ML
repository](http://archive.ics.uci.edu/ml/datasets/Iris)) has 3 classes each
with 50 datapoints each. There are 4 features. The data is preprocessed using
PCA.


    # load data
    #dataset='/home/chiroptera/workspace/datasets/iris/iris.csv'
    dataset='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    irisPCA=True
    normalize=False
    
    irisData=np.genfromtxt(dataset,delimiter=',')
    irisData_o=irisData[:,:-1] # remove classification column
    
    iN,iDims=irisData_o.shape
    
    # PCA of data
    if irisPCA:
        irisData_c,iComps,iEigs=HornAlg.pcaFun(irisData_o,whiten=True,center=True,method='eig',type='corr',normalize=normalize)
        
    #print irisData, nirisData
    #iris true assignment
    irisAssign=np.ones(150)
    irisAssign[50:99]=2
    irisAssign[100:149]=3
    
    #nirisData=sklearn.preprocessing.normalize(nirisData,axis=0)
    
    iFig1=plt.figure()
    plt.title('PCA of iris')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    for i in range(150):
        plt.plot(irisData_c[i,0],irisData_c[i,1],marker='.',c=tableau20[int(irisAssign[i]-1)*2])


![png](Horn_files/Horn_14_0.png)



    print 'Energy of PC (in percentage):'
    print np.cumsum(iEigs)*100/np.sum(iEigs)

    Energy of PC (in percentage):
    [  72.77045209   95.80097536   99.48480732  100.        ]


We choose $\sigma=\frac{1}{4}$ to reproduce the experiments in [3]. We use only
the first two PC here. For more complete results the algorithm is also executed
using all PC.


    sigma=0.25
    steps=80
    
    irisD1,iV1,iE=HornAlg.graddesc(irisData_c[:,0:2],sigma=sigma,steps=steps)


    sigma=0.9
    steps=80
    
    irisD2,iV2,iE=HornAlg.graddesc(irisData_c,sigma=sigma,steps=steps)

## Comments

The results shown above distinguish cluster assignment by colour. However, the
colours might not be consistent throughout all figures. They serve only as a
visual way to see how similar clusters are. This is due to the cluster
assignment algorithm being used. Two methods may be used and differ only in the
order on which they pick points to cluster. They both pick a point from the
clustered data and compute the distance of that point to all the other points.
All the points to which the corresponding distance is below a certain threshold
belong to the same cluster. This process is repeated until all points are
clustered. In one method the point picked to compute the distance is the first
unclustered one. In the other method, the point picked is the one unassigned
point that has the lowest potential value. Both methods suffer from assigning
clusters to outliers.

Before analysing the results, some general comments may be made about the
algorithm. A big advantage of the algorithm is that is does not make assumptions
about the data (number of clusters, shape of clusters, intra-cluster
distribution, etc.). A big disadvantage is that it doesn't assign points to
clusters, if operated in an efficient way, i.e. not computing potential value on
all points but only on datapoints and the direction they take. This is because
the algorithm will converge points toward potential minima (akin to cluster
centers) but will not tell which are these centers, which is the reason that the
assignment methods described above are needed.

## Results
### PC 1 & 2


    dist=1.8
    irisClustering=HornAlg.fineCluster(irisD1,dist)#,potential=iV)
    
    print 'Number of clusters:',max(irisClustering)
    print 'Unclestered points:', np.count_nonzero(irisClustering==0)
    
    iFig2=plt.figure(figsize=(16,12))
    iAx1=iFig2.add_subplot(2,2,1)
    iAx2=iFig2.add_subplot(2,2,2)
    iAx3=iFig2.add_subplot(2,2,4)
    
    iAx1.set_title('Final quantum system')
    iAx1.set_xlabel('PC1')
    iAx1.set_ylabel('PC2')
    
    for i in range(iN):
        if max(irisClustering) >=10:
            c=0
        else:
            c=int(irisClustering[i]-1)*2
        iAx1.plot(irisD1[i,0],irisD1[i,1],marker='.',c=tableau20[c])
    
    iAx2.set_title('Final clustering')
    iAx2.set_xlabel('PC1')
    iAx2.set_ylabel('PC2')
    
    for i in range(iN):
        if max(irisClustering) > 10:
            break
        iAx2.plot(irisData_c[i,0],irisData_c[i,1],marker='.',c=tableau20[int(irisClustering[i]-1)*2])
    
    iAx3.set_title('Original clustering')
    iAx3.set_xlabel('PC1')
    iAx3.set_ylabel('PC2')
    
    for i in range(iN):
        if max(irisClustering) > 10:
            break
        iAx3.plot(irisData_c[i,0],irisData_c[i,1],marker='.',c=tableau20[int(irisAssign[i]-1)*2])
        
    e1=max(map(irisClustering[0:49].tolist().count, irisClustering[0:49]))
    e2=max(map(irisClustering[50:99].tolist().count, irisClustering[50:99]))
    e3=max(map(irisClustering[100:149].tolist().count, irisClustering[100:149]))
    print 'errors=',150-e1-e2-e3

    Number of clusters: 3.0
    Unclestered points: 0
    errors= 22



![png](Horn_files/Horn_21_1.png)




Turning to the results, in the first case (clustering on the 2 first PC), the
results show the clustering algorithm was able to cluster well one of the
clusters (the one that is linearly seperable from the other two) but struggled
with outliers present in the space of the other 2 clusters. Furthermore, the
separation between the yellow and green clusters is hard, which not happens on
the natural clusters. Observing the final quantum system, it's clear that all
points converged to some minima as they are concentrated around some point and
well seperated from other groups of points (other minima). If we were to take
each minima as an independent cluster we would have 11 different clusters, which
is considerably more than the natural 3. This means that some of the minima
might represent micro clusters inside the natural clusters or outliers.

### All PC


    dist=4.5
    irisClustering=HornAlg.fineCluster(irisD2,dist,potential=iV2)
    
    print 'Number of clusters:',max(irisClustering)
    print 'Unclestered points:', np.count_nonzero(irisClustering==0)
    
    iFig2=plt.figure(figsize=(16,12))
    iAx1=iFig2.add_subplot(2,2,1)
    iAx2=iFig2.add_subplot(2,2,2)
    iAx3=iFig2.add_subplot(2,2,4)
    
    iAx1.set_title('Final quantum system')
    iAx1.set_xlabel('PC1')
    iAx1.set_ylabel('PC2')
    
    for i in range(iN):
        if max(irisClustering) >=10:
            c=0
        else:
            c=int(irisClustering[i]-1)*2
        iAx1.plot(irisD2[i,0],irisD2[i,1],marker='.',c=tableau20[c])
    
    iAx2.set_title('Final clustering')
    iAx2.set_xlabel('PC1')
    iAx2.set_ylabel('PC2')
    
    for i in range(iN):
        if max(irisClustering) > 10:
            break
        iAx2.plot(irisData_c[i,0],irisData_c[i,1],marker='.',c=tableau20[int(irisClustering[i]-1)*2])
    
    iAx3.set_title('Original clustering')
    iAx3.set_xlabel('PC1')
    iAx3.set_ylabel('PC2')
    
    for i in range(iN):
        iAx3.plot(irisData_c[i,0],irisData_c[i,1],marker='.',c=tableau20[int(irisAssign[i]-1)*2])
        
    e1=max(map(irisClustering[0:49].tolist().count, irisClustering[0:49]))
    e2=max(map(irisClustering[50:99].tolist().count, irisClustering[50:99]))
    e3=max(map(irisClustering[100:149].tolist().count, irisClustering[100:149]))
    print 'errors=',150-e1-e2-e3

    Number of clusters: 3.0
    Unclestered points: 0
    errors= 27



![png](Horn_files/Horn_23_1.png)


In this case, we use all PC. In the final quantum system, the number of minima
is the same. However, some of the minima are very close to others and have less
datapoints assigned which suggest that they might be local minima and should
probably be annexed to the bigger minima close by. Once again the outliers were
not correctly classified. In this case, though, there is no hard boundary
between the green and yellow clusters. This is due to the fact that we're now
clustering on all PC which bring a greater ammount of information to the problem
(the 2 first PC only ammounted around 95% of the energy).

# Crab

## Preparing dataset

Here we're loading the crab dataset and preprocessing it.


    reload(HornAlg)




    <module 'HornAlg' from 'HornAlg.py'>




    crabsPCA=True
    crabsNormalize=False
    
    crabs=np.genfromtxt('/home/chiroptera/workspace/datasets/crabs/crabs.dat')
    crabsData=crabs[1:,3:]
    
    # PCA
    if crabsPCA:
        ncrabsData1, cComps,cEigs=HornAlg.pcaFun(crabsData,whiten=True,center=False,
                                                 method='eig',type='cov',normalize=crabsNormalize)
        ncrabsData2, cComps,cEigs=HornAlg.pcaFun(crabsData,whiten=True,center=True,
                                                 method='eig',type='corr',normalize=crabsNormalize)
        ncrabsData3, cComps,cEigs=HornAlg.pcaFun(crabsData,whiten=True,center=True,
                                                 method='eig',type='cov',normalize=crabsNormalize)
    
        # real assignment
    crabsAssign=np.ones(200)
    crabsAssign[50:99]=2
    crabsAssign[100:149]=3
    crabsAssign[150:199]=4

We're visualizing the data projected on the second and third principal
components to replicate the results presented on [3]. They use PCA with the
correlation matrix. Below we can see the data on different representations. The
closest representation of the data is using the covariance matrix with
uncentered data (unconventional practice). Using the correlation matrix we get
similar representation to unprocessed data. Although nonconvenional, the
uncentered data plot suggests that data is more seperated that with centered
data, using the covariance matrix.


    cFig1=plt.figure(figsize=(16,12))
    cF1Ax1=cFig1.add_subplot(2,2,1)
    cF1Ax2=cFig1.add_subplot(2,2,2)
    cF1Ax3=cFig1.add_subplot(2,2,3)
    cF1Ax4=cFig1.add_subplot(2,2,4)
    
    cF1Ax1.set_title('Original crab data')
    for i in range(len(crabsAssign)):
        cF1Ax1.plot(crabsData[i,2],crabsData[i,1],marker='.',c=tableau20[int(crabsAssign[i]-1)*2])
    
    cF1Ax2.set_title('Crab projected on PC, Covariance, Uncentered')
    for i in range(len(crabsAssign)):
        cF1Ax2.plot(ncrabsData1[i,2],ncrabsData1[i,1],marker='.',c=tableau20[int(crabsAssign[i]-1)*2])
        
    cF1Ax3.set_title('Crab projected on PC, Correlation, Centered')
    for i in range(len(crabsAssign)):
        cF1Ax3.plot(ncrabsData2[i,2],ncrabsData2[i,1],marker='.',c=tableau20[int(crabsAssign[i]-1)*2])
        
    cF1Ax4.set_title('Crab projected on PC, Covariance, Centered')
    for i in range(len(crabsAssign)):
        cF1Ax4.plot(ncrabsData3[i,2],ncrabsData3[i,1],marker='.',c=tableau20[int(crabsAssign[i]-1)*2])


![png](Horn_files/Horn_29_0.png)


## Cluster

We're clustering according to the second and third PC to try to replicate [3],
along with the same $\sigma$.


    sigma=1.0/sqrt(2)
    steps=80
    crab2cluster=ncrabsData1
    crabD,V,E=HornAlg.graddesc(crab2cluster[:,1:3],sigma=sigma,steps=steps)


    dist=1
    crabClustering=HornAlg.fineCluster(crabD,dist,potential=V)
    
    print 'Number of clusters:',max(crabClustering)
    print 'Unclestered points:', np.count_nonzero(crabClustering==0)
    
    cFig2=plt.figure(figsize=(16,12))
    cAx1=cFig2.add_subplot(2,2,1)
    cAx2=cFig2.add_subplot(2,2,3)
    cAx3=cFig2.add_subplot(2,2,4)
    #cFig2,(cAx1,cAx2)=plt.subplots(nrows=1, ncols=2, )
    
    cAx1.set_title('Final quantum system')
    for i in range(len(crabsAssign)):
        if max(crabClustering) >= 10:
            c=0
        else:
            c=int(crabClustering[i]-1)*2
        cAx1.plot(crabD[i,0],crabD[i,1],marker='.',c=tableau20[c])
    
    cAx2.set_title('Final clustering')
    for i in range(len(crabsAssign)):
        if max(crabClustering) > 10:
            break
        cAx2.plot(crab2cluster[i,2],crab2cluster[i,1],marker='.',c=tableau20[int(crabClustering[i]-1)*2])
        
    cAx3.set_title('Original clustering')
    for i in range(len(crabsAssign)):
        cAx3.plot(crab2cluster[i,2],crab2cluster[i,1],marker='.',c=tableau20[int(crabsAssign[i]-1)*2])

    Number of clusters: 4.0
    Unclestered points: 0



![png](Horn_files/Horn_32_1.png)


The 'Final quantum system' shows how the points evolved in 80 steps. We can see
that they all converged to 4 minima of the potential for
$\sigma=\frac{1}{\sqrt{2}}$, making it easy to identify the number of clusters
to choose. However, this is only clear observing the results. The distance used
to actually assign the points to the clusters need tampering with a per problem
basis. We can see that outliers usually were incorrectly clustered. Plus, a
considerable portion of data was also wrongly clustered. The accuracy of the
clustering was the following:


    c1=max(map(crabClustering[0:49].tolist().count, crabClustering[0:49]))
    c2=max(map(crabClustering[50:99].tolist().count, crabClustering[50:99]))
    c3=max(map(crabClustering[100:149].tolist().count, crabClustering[100:149]))
    c4=max(map(crabClustering[150:199].tolist().count, crabClustering[150:199]))
    print 'Errors on cluster 1:\t',50-c1
    print 'Errors on cluster 2:\t',50-c2
    print 'Errors on cluster 3:\t',50-c3
    print 'Errors on cluster 4:\t',50-c4
    print 'errors=',200-c1-c2-c3-c4

    Errors on cluster 1:	20
    Errors on cluster 2:	1
    Errors on cluster 3:	16
    Errors on cluster 4:	1
    errors= 38


## Conventional PCA

Now we'll cluster with the conventional PCA, with the correlation matrix.


    sigma=1.0/sqrt(2)
    steps=80
    crab2cluster=ncrabsData3
    crabD,V,E=HornAlg.graddesc(crab2cluster[:,1:3],sigma=sigma,steps=steps)


    dist=1
    crabClustering=HornAlg.fineCluster(crabD,dist,potential=V)
    
    print 'Number of clusters:',max(crabClustering)
    print 'Unclestered points:', np.count_nonzero(crabClustering==0)
    
    cFig2=plt.figure(figsize=(16,12))
    cAx1=cFig2.add_subplot(2,2,1)
    cAx2=cFig2.add_subplot(2,2,3)
    cAx3=cFig2.add_subplot(2,2,4)
    #cFig2,(cAx1,cAx2)=plt.subplots(nrows=1, ncols=2, )
    
    cAx1.set_title('Final quantum system')
    for i in range(len(crabsAssign)):
        if max(crabClustering) >= 10:
            c=0
        else:
            c=int(crabClustering[i]-1)*2
        cAx1.plot(crabD[i,0],crabD[i,1],marker='.',c=tableau20[c])
    
    cAx2.set_title('Final clustering')
    for i in range(len(crabsAssign)):
        if max(crabClustering) > 10:
            break
        cAx2.plot(crab2cluster[i,2],crab2cluster[i,1],marker='.',c=tableau20[int(crabClustering[i]-1)*2])
        
    cAx3.set_title('Original clustering')
    for i in range(len(crabsAssign)):
        cAx3.plot(crab2cluster[i,2],crab2cluster[i,1],marker='.',c=tableau20[int(crabsAssign[i]-1)*2])

    Number of clusters: 4.0
    Unclestered points: 0



![png](Horn_files/Horn_37_1.png)



    c1=max(map(crabClustering[0:49].tolist().count, crabClustering[0:49]))
    c2=max(map(crabClustering[50:99].tolist().count, crabClustering[50:99]))
    c3=max(map(crabClustering[100:149].tolist().count, crabClustering[100:149]))
    c4=max(map(crabClustering[150:199].tolist().count, crabClustering[150:199]))
    print 'Errors on cluster 1:\t',50-c1
    print 'Errors on cluster 2:\t',50-c2
    print 'Errors on cluster 3:\t',50-c3
    print 'Errors on cluster 4:\t',50-c4
    print 'errors=',200-c1-c2-c3-c4

    Errors on cluster 1:	11
    Errors on cluster 2:	2
    Errors on cluster 3:	1
    Errors on cluster 4:	5
    errors= 19


Using conventional PCA, clustering results are better.

## Other preprocessing
Let's now consider clustering on data projected on all principal components
(with centered data) and on original data.


    #1.0/np.sqrt(2)
    sigma_allpc=0.5
    steps_allpc=200
    crabD_allpc,V_allpc,E=HornAlg.graddesc(ncrabsData3[:,:3],sigma=sigma_allpc,steps=steps_allpc)


    sigma_origin=1.0/sqrt(2)
    steps_origin=80
    crabD_origin,V_origin,E=HornAlg.graddesc(crabsData,sigma=sigma_origin,steps=steps_origin)


    dist_allpc=12
    dist_origin=25
    
    crabClustering_allpc=HornAlg.fineCluster(crabD_allpc,dist_allpc,potential=V_allpc)
    crabClustering_origin=HornAlg.fineCluster(crabD_origin,dist_origin,potential=V_origin)
    
    print 'All PC\t\tNumber of clusters:',max(crabClustering_allpc)
    print 'All PC\t\tUnclestered points:', np.count_nonzero(crabClustering_allpc==0)
    print 'Original data\tNumber of clusters:',max(crabClustering_origin)
    print 'Original data\tUnclestered points:', np.count_nonzero(crabClustering_origin==0)
    
    cFig2=plt.figure(figsize=(16,12))
    cAx1=cFig2.add_subplot(3,2,1)
    cAx2=cFig2.add_subplot(3,2,3)
    cAx3=cFig2.add_subplot(3,2,5)
    
    cAx4=cFig2.add_subplot(3,2,2)
    cAx5=cFig2.add_subplot(3,2,4)
    cAx6=cFig2.add_subplot(3,2,6)
    
    cAx1.set_title('Final quantum system, All PC')
    for i in range(len(crabsAssign)):
        if max(crabClustering_allpc) >= 10:
            c=0
        else:
            c=int(crabClustering_allpc[i]-1)*2
        cAx1.plot(crabD_allpc[i,1],crabD_allpc[i,2],marker='.',c=tableau20[c])
    
    cAx2.set_title('Final clustering, All PC')
    for i in range(len(crabsAssign)):
        if max(crabClustering_allpc) > 10:
            break
        cAx2.plot(ncrabsData3[i,1],ncrabsData3[i,2],marker='.',c=tableau20[int(crabClustering_allpc[i]-1)*2])
        
    cAx3.set_title('Original clustering, All PC')
    for i in range(len(crabsAssign)):
        cAx3.plot(ncrabsData3[i,1],ncrabsData3[i,2],marker='.',c=tableau20[int(crabsAssign[i]-1)*2])
    
        #--------------------------------------------------------------#
        
    cAx4.set_title('Final quantum system, Original data')
    for i in range(len(crabsAssign)):
        if max(crabClustering_origin) >= 10:
            c=0
        else:
            c=int(crabClustering_origin[i]-1)*2
        cAx4.plot(crabD_origin[i,0],crabD_origin[i,1],marker='.',c=tableau20[c])
    
    cAx5.set_title('Final clustering, Original Data')
    for i in range(len(crabsAssign)):
        if max(crabClustering_origin) > 10:
            break
        cAx5.plot(crabsData[i,0],crabsData[i,1],marker='.',c=tableau20[int(crabClustering_origin[i]-1)*2])
        
    cAx6.set_title('Original clustering, Original Data')
    for i in range(len(crabsAssign)):
        cAx6.plot(crabsData[i,0],crabsData[i,1],marker='.',c=tableau20[int(crabsAssign[i]-1)*2])

    All PC		Number of clusters: 3.0
    All PC		Unclestered points: 0
    Original data	Number of clusters: 2.0
    Original data	Unclestered points: 0



![png](Horn_files/Horn_43_1.png)


The results of the last experimens show considerably worse results. The final
quantum system suggests a great ammount of minima and bigger variance on the
final convergence of the points. Furthermore the distribution of the minima
doesn't suggest any natural clustering for the user, contrary to what happened
before.

The clustering on raw data is very bad. This was to be expected considering the
distribution and shape of the original data across all dimensions and clusters.

# Gaussian blobs

## Original Mix


    n_samples=400
    n_features=5
    centers=4
    
    x_Gauss,x_assign=sklearn.datasets.make_blobs(n_samples=n_samples,n_features=n_features,centers=centers)
    #nX=sklearn.preprocessing.normalize(x_Gauss,axis=0)
    x_2cluster=x_Gauss
    
    gMix_fig=plt.figure()
    plt.title('Gaussian Mix, '+str(n_features)+' features')
    for i in range(x_Gauss.shape[0]):
        plt.plot(x_2cluster[i,0],x_2cluster[i,1],marker='.',c=tableau20[int(x_assign[i])*2])


![png](Horn_files/Horn_47_0.png)



    sigma=2.
    steps=200
    gaussD,V,E=HornAlg.graddesc(x_2cluster,sigma=sigma,steps=steps)


    dist=6
    nX_clustering=HornAlg.fineCluster(gaussD,dist,potential=V)
    print 'number of clusters=',max(nX_clustering)
    
    gRes_fig=plt.figure(figsize=(16,12))
    gRes_ax1=gRes_fig.add_subplot(2,2,1)
    gRes_ax2=gRes_fig.add_subplot(2,2,2)
    gRes_ax3=gRes_fig.add_subplot(2,2,4)
    
    
    gRes_ax1.set_title('Final quantum system')
    for i in range(x_Gauss.shape[0]):
        if max(nX_clustering) > 10:
            c=0
        else:
            c=int(nX_clustering[i]-1)*2
        gRes_ax1.plot(gaussD[i,0],gaussD[i,1],marker='.',c=tableau20[c])
    
    gRes_ax2.set_title('Final clustering')
    for i in range(len(nX_clustering)):
        if max(nX_clustering) >10 :
            break
        gRes_ax2.plot(x_2cluster[i,0],x_2cluster[i,1],marker='.',c=tableau20[int(nX_clustering[i]-1)*2])
        
    gRes_ax3.set_title('Correct clustering')
    for i in range(x_Gauss.shape[0]):
        gRes_ax3.plot(x_2cluster[i,0],x_2cluster[i,1],marker='.',c=tableau20[int(x_assign[i])*2])

    number of clusters= 112.0



![png](Horn_files/Horn_49_1.png)


## PCA Mix


    pcaX,gaussComps,gaussEigs=HornAlg.pcaFun(x_Gauss,whiten=True,center=True,
                                                 method='eig',type='cov',normalize=False)
    gPCAf=plt.figure()
    plt.title('PCA')
    for i in range(x_Gauss.shape[0]):
        plt.plot(pcaX[i,0],pcaX[i,1],marker='.',c=tableau20[int(x_assign[i])*2])


![png](Horn_files/Horn_51_0.png)



    sigma=2.
    steps=400
    pcaGaussD,V,E,eta=HornAlg.graddesc(pcaX,sigma=sigma,steps=steps,return_eta=True)


    dist=28
    pcaX_clustering=HornAlg.fineCluster(pcaGaussD,dist,potential=V)
    print 'number of clusters=',max(pcaX_clustering)
    
    gPCARes_fig,(gPCARes_ax1,gPCARes_ax2)=plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    
    gPCARes_ax1.set_title('Final quantum system')
    for i in range(x_Gauss.shape[0]):
        if max(pcaX_clustering) > 10:
            c=0
        else:
            c=int(pcaX_clustering[i]-1)*2
        gPCARes_ax1.plot(pcaGaussD[i,0],pcaGaussD[i,2],marker='.',c=tableau20[c])
    
    gPCARes_ax2.set_title('Final clustering')
    for i in range(len(pcaX_clustering)):
        if max(pcaX_clustering) >10 :
            break
        gPCARes_ax2.plot(pcaX[i,0],pcaX[i,1],marker='.',c=tableau20[int(pcaX_clustering[i]-1)*2])

    number of clusters= 4.0



![png](Horn_files/Horn_53_1.png)


## Comments

The algorithm performed very poorly in unprocessed data. Even with a high
$$\sigma$$ and big number of steps for the gradient descent, the final quantum
system had points scattered all over, not even seemingly alike the original
data, i.e. the points diverged. The performance on the projected data was
significantly better. The final quantum system suggests some natural clustering
to a user, assimilating 4 seperate clusters. However the assignment algorithm
did a very poor job and the final clustering is all off. Paying close attention
to the colours, though, we can see that the left most cluster only has two
colours in both plots, which suggest a correspondence. The same can be done to
the other clusters. A user using this algorithm would be able to do a better
clustering in selecting which points should be together by analyzing the final
quantum system plot. The assignment algorithm probably performs worse because of
other dimensions.
