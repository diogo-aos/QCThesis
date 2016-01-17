# Survey of ensemble algorithms

https://www.researchgate.net/profile/Sandro_Vega-Pons/publication/220360297_A_Survey_of_Clustering_Ensemble_Algorithms/links/0912f50b54efa3d841000000.pdf

This article presents a survey of ensemble algorithms up to 2011 (?). It does not present results but it does include EAC and other co-association matrix based algorithms, stating that those are not scalable. I could reference this paper commenting that our work challenges this belief.

They state that the general class of algorithms that EAC belongs cannot be applied to large datasets because of quadratic complexity. However, our implementation has a reduced complexity due to the Kruskal's algorithm which has a ElogV complexity. In our results, we observe that the number of associations (E) is significantly inferior to n^2, which permits the application of the algorithm to larger datasets that would otherwise be possible.

# light reference to objective functions use in large datasets

http://www.crpit.com/confpapers/CRPITV101Yearwood.pdf

this paper uses 4 consensus functions (from http://ir.library.oregonstate.edu/xmlui/bitstream/handle/1957/35655/2006-23.pdf?sequence=1) stating "in terms of their application most work to date has been done on cluster ensembling of large datasets", but their application in the source article examines only small datasets...


# ensemble framework for large datasets

(access with credentials)
http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4274398

This paper proposes a scalable cluster ensemble framework for large datasets (SBEC), but only shows results to Iris (150,4), ISOLET (1440,617) and Digits (3498,16) - not exactly what we aim for large datasets.

According to this paper, SCEC has a time complexity of O(tnr^2k), where t is the number of iterations and r is the number of clusters in the ensemble partitions; CSPA: O(n^2kr); MCLA: O(nk^2r^2); it states that QMI and MMEC have the same complexity as SCEC. These are all complexities post ensemble creation. SBEC has complexity of O(rk^2d + rk^3 + kr^2). The difference between r and k is that r is the #clusters in ensemble partitions and k is the #clusters final solution.

After ensemble creation, EAC's biggest complexity is the MST ElogN, where E was observed to be <<N^2 but >N. As such, SBEC is significantly faster than EAC and any other since it doesn't scale with N, but with the number of clusters in the partitions of the ensemble. EAC is for sure faster than CSPA, hard to compare with MCLA and SCEC, but I'd say it has a good chance of being faster. Furthermore, I don't know what kind of base algorithms the ensemble was produced from. If those other algorithms need more complex algorithms than K-Means to produce the ensemble and yield good results, then EAC, in the end, might win because we're parallelizing on the GPU.

I might take this route as an alternative to actually running the algorithms. I couldn't find implementation of any of these algorithms, although SBEC should be easy to implement since I've already been using the Hungarian algorithm (it's their tool for merging the ensemble).