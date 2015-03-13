# Questions to discuss

Talk about the quantum clustering overview.

Point out that, using the qubit paradigm, any quantum analog of a classical clustering algorithm will be slower.

Point out that most qubit approaches are implemented using algorithms already inspired on other physical phenomena, e.g. genetic algorithms, hives.

Talk about the dynamic quantum clustering. It seems to be a goot tool for visual exploration, as one paper suggests, but very hard to use as standalone without user intervention. The main reason for that is that in order for the algorithm to be minimally efficient it cannot compute the potential function on all space. This translates in deficient cluster assignment since we don't know the cluster centers, i.e. the minima of the potential function - we only know where the points descended to.

All the articles about quantum computing/clustering point to the use of quantum oracles, which in turn point to the use of quantum computers/circuits. The one exception was an article given by Helena which describes several quantum inspired methods. These are implemented with a probabilistic Turing machine (typically uniform distribution on machine alphabet). These are:
- quantum k-means with genetic operations
- fuzzy C-means with genetic operations
- 

# cluster ensembles


