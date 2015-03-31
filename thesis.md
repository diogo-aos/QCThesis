---
title: Speeding up clustering ensembles
author: Diogo Silva
---

# Structure
The scope of the thesis is Big Data and Cluster Ensembles. To both, a main requirement is to have fast clustering techniques. This may be accomplished in two ways: algorithmically or with parallelization techniques. The former deals with finding faster solutions while the later takes existing solutions and optimizes them having execution speed in mind.

The initial research was under the algorithmic path. More specifically, exploring quantum clustering algorithms. The findings of this exploration were unproductive and turned the focus of the research to parallelization techniques. Two main paradigms of parallelization were found: GPU and distributed (among several machines).

# Quantum Clustering

There are two major paths for the problem of quantum clustering. The first is the quantization of clustering methods to work in quantum computers. This is basically converting algorithms to work partially or totally on a different computing paradigm, with support of quantum circuits or computers. Literature suggests that quadratic (and even exponential in some cases) speedup may be achieved. Most of the approaches for such conversions make use of Groover's search algorithm, or a variant of it, e.g. [1]. Most literature on this path is also quite theoretical since quantum computers or quantum circuits are not easily available. This path can be seen as part of the bigger problem of quantum computing and quantum information processing. 

The second path is the computational intelligence approach, i.e.  to use quantum inspired algorithms that muster inspiration from quantum analogies. A study of the literature will reveal that this path typically further divides itself into two other branches. One comprehends the algorithms based on the concept of the quantum bit, the quantum analogue of a classical bit with interesting properties found in quantum objects. The other branch models data as a quantum system and uses the Schrödinger equation to evolve it.

## Quantum bit

The quantum bit is a quantum object that has the properties of quantum superposition, entanglement and ...

Introduce the concept further

Several clustering algorithms [4-6], as well as optimization problems [7], are modelled after this concept. To test the potential of the algorithms under this paradigm, a quantum variant of the K-Means algorithm [5] was chosen as a case study.

## Quantum K-Means
### Description of the algorithm

The Quantum K-Means algorithm, as is described in [5], is based on the classical K-Means algorithm. It extends the basic K-Means with concepts from quantum mechanics (the qubit) and genetic algorithms.

(describe algorithm...)


The algorithm implemented and tested is a variant of the one described in [5]. The genetic operations of cross-over and mutation are taken away. This decision was based on the findings of [8], stating that the use of the angle-distance rotation method in the quantum rotation operation produces enough variability.

### Testing 
The testing was aimed at benchmarking both accuracy and speed. The input used was synthetic data, namely, Gaussian mixtures with variable cardinality and dimensionality. The results

(copy of report)

Regarding the Quantum K-Means (QK-Means), the tests were performed using 10 oracles, a qubit string length of 8 and 100 generations per round. The \textit{classical} K-Means was executed using the \textit{k-means++} centroid initialization method. Since QK-Means executes a classical K-Means for each oracle each generation, the number of initializations for K-Means was $\#oracles \times \#generations \times factor$, where $factor$ is a adjustable multiplier. Each test had 20 rounds.

All tests were done with 6 clusters (natural number of clusters). Two tests were done with the two dimensional dataset: one with a $factor=1.10$ (increase initializations by 10\%) and another with $factor=1$. I'll call these tests T1 and T2. The test done with the six dimensional dataset (T3) used $factor=1.10$.

### Results

#### Timing results

(table)

The mean computation time of classical K-Means is an order of magnitude lower than that of QK-Means. However, in classical K-Means the solution typically chosen if the one with lowest sum of euclidean distances of points to their attributed centroid. To make a fair comparisson between the two algorithms, the Davies-Bouldin index of all classical K-Means solutions was computed and used as the criterea to choose the best solution. When this is done, we can see that the total time of classical K-Means is actually higher that that of QK-Means in T1 and T3, but this is only due to the 1.10 multiplier on the number of initializations. In T2, possibly the fairest comparisson, the computation times become very similar with only a 2\% difference between the two algorithms.


## Schrödinger equation
The first step in this methodology is to compute a probability density function of the input data. This is done with a Parzen-window estimator in [2,3] This function will be the wave function in the Schrodinger equation. Having this information we'll compute the potential function that corresponds to the state of minimum energy (ground state = eigenstate with minimum eigenvalue) [2].

This potential function is almost like the inverse of a probability density function. Minima of the potential correspond to intervals in space where points are together. So minima will naturally correspond to cluster centers [2]. The computation of the potential function on every point in space is a costly computation effort though, so what is done in one method is to compute the potential on the input data and converge this points toward the minima in the potential function. This is done with the gradient descent method in [2]. Another method [3] is to think of the input data as particles and use Hamiltonian operator to evolve the quantum system in the time-dependant Schrodinger equation. Given enough time steps, the particles will oscilate around potential minima.

Both methods take as input parameter the variance of the parzen-window estimator $$sigma$$.

# References
[1] N. Wiebe, A. Kapoor, and K. Svore, “Quantum Algorithms for Nearest-Neighbor Methods for Supervised and Unsupervised Learning,” p. 31, 2014.
[2] D. Horn and A. Gottlieb, “The Method of Quantum Clustering.,” NIPS, no. 1, 2001.
[3] M. Weinstein and D. Horn, “Dynamic quantum clustering: a method for visual exploration of structures in data,” Phys. Rev. E - Stat. Nonlinear, Soft Matter Phys., vol. 80, no. 6, pp. 1–15, Dec. 2009.


[4] E. Casper and C. Hung, “Quantum Modeled Clustering Algorithms for Image Segmentation,” vol. 2, no. March, pp. 1–21, 2013.
[5] E. Casper, C.-C. Hung, E. Jung, and M. Yang, “A Quantum-Modeled K-Means Clustering Algorithm for Multi-band Image Segmentation.” [Online]. Available: http://delivery.acm.org/10.1145/2410000/2401639/p158-casper.pdf?ip=193.136.132.10&id=2401639&acc=ACTIVE SERVICE&key=2E5699D25B4FE09E.F7A57B2C5B227641.4D4702B0C3E38B35.4D4702B0C3E38B35&CFID=476955365&CFTOKEN=55494231&__acm__=1423057410_0d77d9b5028cb3. [Accessed: 04-Feb-2015].
[6] J. Xiao, Y. Yan, J. Zhang, and Y. Tang, “A quantum-inspired genetic algorithm for k-means clustering,” Expert Syst. Appl., vol. 37, pp. 4966–4973, 2010.

[7] H. Wang, J. Liu, J. Zhi, and C. Fu, “The Improvement of Quantum Genetic Algorithm and Its Application on Function Optimization,” vol. 2013, no. 1, 2013.

[8] W. Liu, H. Chen, Q. Yan, Z. Liu, J. Xu, and Y. Zheng, “A novel quantum-inspired evolutionary algorithm based on variable angle-distance rotation,” 2010 IEEE World Congr. Comput. Intell. WCCI 2010 - 2010 IEEE Congr. Evol. Comput. CEC 2010, 2010.