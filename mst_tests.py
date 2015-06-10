# -*- coding: utf-8 -*-

"""
Tests for MST algorithms.
"""

import numpy as np
from numba import cuda, jit
from timeit import default_timer as timer
from scipy.sparse.csr import csr_matrix
from MyML.graph.mst import boruvka_minho_seq, boruvka_minho_gpu

import sys



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                     UTILS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Timer():
    def __init__(self):
        self.start = 0
        self.end = 0

    def tic(self):
        self.start = timer()

    def tac(self):
        self.end = timer()
        self.elapsed = self.end - self.start
        return self.elapsed

@jit
def outdegree_from_firstedge(firstedge, outdegree, n_edges):
    n_vertices = firstedge.size

    for v in range(n_vertices - 1):
        outdegree[v] = firstedge[v + 1] - firstedge[v]

    outdegree[n_vertices - 1] = n_edges - firstedge[n_vertices - 1]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                     GRAPHS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # 

simple_graph = dict()
simple_graph["dest"] = np.array([1, 3, 2, 0, 3, 0, 3, 0, 1, 2, 5, 4, 6, 7, 5, 7, 6, 7], dtype = np.int32)
simple_graph["weight"] = np.array([2, 2, 1, 2, 3, 1, 3, 2, 3, 3, 1, 1, 3, 7, 3, 2, 2, 7], dtype = np.float32)
simple_graph["firstEdge"] = np.array([0, 3, 5, 7, 10, 11, 14, 16], dtype = np.int32)
simple_graph["outDegree"] = np.array([3, 2, 2, 3, 1, 3, 2, 2], dtype = np.int32)

# # # # # # # # # # # # # # # # # # 
simple_graph_connect = dict()
simple_graph_connect["dest"] = np.array([1, 2, 3, 2, 0, 2, 0, 1, 7, 5, 4, 6, 7, 5, 7, 3, 5, 6], dtype = np.int32)
simple_graph_connect["weight"] = np.array([3, 1, 2, 1, 2, 3, 2, 3, 3, 1, 1, 3, 7, 3, 2, 3, 7, 2], dtype = np.float32)
simple_graph_connect["firstEdge"] = np.array([0, 3, 4, 6, 9, 10, 13, 15], dtype = np.int32)
simple_graph_connect["outDegree"] = np.array([3, 1, 2, 3, 1, 3, 2, 3], dtype = np.int32)

# # # # # # # # # # # # # # # # # # 

four_elt_mat = np.genfromtxt("4elt.edges", delimiter=" ",
                              dtype=[("firstedge","i4"),("dest","i4"),("weight","f4")],
                              skip_header=1)
four_elt_mat_s = csr_matrix((four_elt_mat["weight"], (four_elt_mat["firstedge"], four_elt_mat["dest"])))

del four_elt_mat

four_elt = dict()
four_elt["dest"] = four_elt_mat_s.indices
four_elt["weight"] = four_elt_mat_s.data
four_elt["firstEdge"] = four_elt_mat_s.indptr
four_elt["outDegree"] = np.empty_like(four_elt["firstEdge"])

del four_elt_mat_s

outdegree_from_firstedge(four_elt["firstEdge"], four_elt["outDegree"], four_elt["dest"].size)

# # # # # # # # # # # # # # # # # # 

def load_graph(name):
    # simple graph of 8 vertices and 9 edges

    graph_names = {"simple_graph" : simple_graph,
                   "simple_graph_connect" : simple_graph_connect,
                   "4elt" : four_elt}

    if name not in graph_names.keys():
        raise Exception("GRAPH " + name + " DOES NOT EXIST.")
    else:
        graph = graph_names[name]

        return graph["dest"], graph["weight"], graph["firstEdge"], graph["outDegree"]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                     TESTS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def host_boruvka():

    print "HOST CPU BORUVKA"

    dest, weight, firstEdge, outDegree = load_graph("simple_graph_connect")

    t1 = Timer()
    t1.tic()
    mst, n_edges = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
    t1.tac()

    print "time elapsed: ", t1.elapsed
    mst.sort() # mst has to be sorted for comparison with device mst because different threads might be able to write first
    print mst
    print n_edges

def device_boruvka():

    print "CUDA BORUVKA"

    dest, weight, firstEdge, outDegree = load_graph("simple_graph_connect")

    t1 = Timer()
    t1.tic()
    mst, n_edges = boruvka_minho_gpu(dest, weight, firstEdge, outDegree)
    t1.tac()

    print "time elapsed: ", t1.elapsed
    mst.sort()
    print mst
    print n_edges

def host_vs_device():
    print "HOST VS DEVICE"

    dest, weight, firstEdge, outDegree = load_graph("simple_graph_connect")

    t1, t2 = Timer(), Timer()

    t1.tic()
    mst1, n_edges1 = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
    t1.tac()


    t2.tic()
    mst2, n_edges2 = boruvka_minho_gpu(dest, weight, firstEdge, outDegree)
    t2.tac()

    print "time elapsed: ", t1.elapsed
    print mst1
    print n_edges



def main(argv):
    valid_args = [0, 1, 2]
    valid_args = map(str,valid_args)
    if len(argv) <= 1 or argv[1] not in valid_args:
        print "0 : test host boruvka"
        print "1 : test device boruvka"

    elif argv[1] == "0":
        host_boruvka()
    elif argv[1] == "1":
        device_boruvka()
    elif argv[1] == "2":
        host_vs_device()        
        


if __name__ == "__main__":
    main(sys.argv)