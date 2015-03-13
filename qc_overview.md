# Quantum Clustering overview

There are two major paths for the problem of quantum Clustering. The first is the quantization of clustering methods to work in quantum computers. This is basically converting algorithms to work partially or totally on a different paradigm. Literature suggests that quadratic (and even exponential in some cases) speedup may be achieved. Most of the approaches for such conversions make use of Groover's search algorithm, or a variant of it, e.g. [1]. Most literature on this path is also quite theoretical since quantum computers or quantum circuits are not easily available. This path can be seen as part of the bigger problem of quantum computing and quantum information processing. 

The second path is to use quantum inspired algorithms, i.e. methods that muster inspiration from quantum analogies. A study of the literature will reveal that this path further divides itself into two other branches. One compreends the algorithms based on the concept of the quantum bit, the quantum analog of a classical bit with interesting properties found in quantum objects. The other branch uses the schodinger equation. 

## What is the quantum bit

The quantum bit is a quantum object that has the properties of quantum superposition, entanglement and 

## How do they use the schodinger equation? 
The first step in this methodology is to compute a probability density function of the input data. This is done with a Parzen-window estimator in [2,3] This function will be the wave function in the Schrodinger equation. Having this information we'll compute the potential function that corresponds to the state of minimum energy (ground state = eigenstate with minimum eigenvalue) [2].

This potential function is almost like the inverse of a probability density function. Minima of the potential correspond to intervals in space where points are together. So minima will naturally correspond to cluster centers [2]. The computation of the potential function on every point in space is a costly computation effort though, so what is done in one method is to compute the potential on the input data and converge this points toward the minima in the potential function. This is done with the gradient descent method in [2]. Another method [3] is to think of the input data as particles and use Hamiltonian operator to evolve the quantum system in the time-dependant Schrodinger equation. Given enough time steps, the particles will oscilate around potential minima.

Both methods take as input parameter the variance of the parzen-window estimator $$sigma$$.


## References
[1] N. Wiebe, A. Kapoor, and K. Svore, “Quantum Algorithms for Nearest-Neighbor Methods for Supervised and Unsupervised Learning,” p. 31, 2014.
[2] D. Horn and A. Gottlieb, “The Method of Quantum Clustering.,” NIPS, no. 1, 2001.
[3] M. Weinstein and D. Horn, “Dynamic quantum clustering: a method for visual exploration of structures in data,” Phys. Rev. E - Stat. Nonlinear, Soft Matter Phys., vol. 80, no. 6, pp. 1–15, Dec. 2009.
