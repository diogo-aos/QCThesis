title: Evidence Accumulation Clustering for Big Data on GPU with CUDA
authors: Diogo Silva


- shortcomings of original EAC (ensemble generation(?),square coassocs, etc.)
- how I solved them (the real contribution: speed-up ensemble generation, NxK coassocs, K-Means vs Single-Link)
- compare accuracy from original EAC vs big data one
- compare performance
- talk about the tradeoff between the two

# Abstract

# Introduction
The scope of this work is the study of the viability of using evidence accumulation 

Combining multiple clusterings using evidence accumulation has been applied with success to difficult data [Fred, Jain, 2005]. This method depends upon multiple clusterings over the data. Clustering techniques capable of yielding the true structure of unstructured data are of high value to the paradigm of Big Data. However, machine learning techniques applied to this realm should be fast in order to be feasible. The scope of this work is to study the feasibility of applying this method to Big Data. More specifically, harnessing the power of general-purpose computing on graphics processing units and the flexibility of the Compute Unified Device Architecture (CUDA) programming model.

<!-- short literature review of previous success cases on the application of algorithms in parallel computational models (emphasis on GPU) -->



# CUDA
## brief explanaton of the programming model



# K-Means
## Brief description of the algorithm
<!-- pseudocode of sequential version -->

## parallel version

<!-- pseudocode of parallel version -->

con straints on parallelization
-  sequential data dependency
-  memory bounded


# Evidence accumulation
## Brief description of the algorithm


# Results

## Speeding up K-Means

## Speeding up coassoc matrix

# Conclusion

# References

[1] M. Zechner and M. Granitzer, “Accelerating k-means on the graphics processor via CUDA,” Proc. 1st Int. Conf. Intensive Appl. Serv. INTENSIVE 2009, pp. 7–15, 2009.
[2] J. DiMarco and M. Taufer, “Performance impact of dynamic parallelism on different clustering algorithms,” Spie, p. 87520E, 2013.
[3] H. T. Bai, L. L. He, D. T. Ouyang, Z. S. Li, and H. Li, “K-means on commodity GPUs with CUDA,” 2009 WRI World Congr. Comput. Sci. Inf. Eng. CSIE 2009, vol. 3, pp. 651–655, 2009.
[4] A. N. L. Fred and A. K. Jain, “Combining multiple clusterungs using evidence accumulation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 27, no. 6, pp. 835–850, 2005.