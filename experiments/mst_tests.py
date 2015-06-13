# -*- coding: utf-8 -*-

"""
Tests for MST algorithms.
"""

import numpy as np
from numba import cuda, jit, int32, float32
from timeit import default_timer as timer
from scipy.sparse.csr import csr_matrix
from MyML.graph.mst import boruvka_minho_seq, boruvka_minho_gpu
from MyML.graph.connected_components import connected_comps_seq as getLabels
from MyML.helper.scan import exprefixsumNumba

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


def special_bfs(dest, fe, od, mst):

    undiscovered = set(range(fe.size))
    queue = [0]
    n_mst = 1

    while len(undiscovered) != 0:
        vertex = queue.pop()
        
        start, end = fe[vertex], fe[vertex] + od[vertex]
        for edge in range(start, end):
            dest_vertex = dest[edge]
            if dest_vertex not in discovered:
                queue.append(dest_vertex)
                undiscovered.remove(dest_vertex)

        if len(queue) == 0 and len(undiscovered) != 0:
            queue.append(undiscovered.pop())
            n_mst += 1

    return n_mst

@cuda.jit
def newOutDegree(mst, dest, fe):
    v = cuda.grid(1)
    pass

@jit
def binaryEdgeIdSearch(key, dest, fe, od):
    imin = 0
    imax = fe.size

    while imin < imax:
        imid = (imax + imin) / 2

        imid_fe = fe[imid]
        # key is before
        if key < imid_fe:
            imax = imid
        # key is after
        elif key > imid_fe + od[imid] - 1:
            imin = imid + 1
        # key is between first edge of imid and next first edge
        else:
            return imid
    return -1

@jit(["int32(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:])"], nopython=True)
def get_new_graph(dest, weight, fe, od, mst, nod, nfe, ndest, nweight):

    # first build the outDegree to get the first_edge
    for e in range(mst.size):
        edge = mst[e]
        o_v = dest[edge] # destination
        i_v = binaryEdgeIdSearch(edge, dest, fe, od)
        if i_v == -1:
            return -1
        nod[o_v] += 1
        nod[i_v] += 1

    # get first edge from outDegree
    exprefixsumNumba(nod, nfe, init = 0)

    # get copy of newFirstEdge to serve as pointers for the newDest
    top_edge = np.empty(nfe.size, dtype = int32)
    for i in range(nfe.size):
        top_edge[i] = nfe[i]

    # go through all the mst edges again and write the new edges in the new arrays
    for e in range(mst.size):
        edge = mst[e]

        o_v = dest[edge] # destination vertex
        i_v = binaryEdgeIdSearch(edge, dest, fe, od)
        if i_v == -1:
            return -1
        
        i_ptr = top_edge[i_v]
        o_ptr = top_edge[o_v]

        ndest[i_ptr] = o_v
        ndest[o_ptr] = i_v

        edge_w = weight[edge]
        nweight[i_ptr] = edge_w
        nweight[o_ptr] = edge_w

        top_edge[i_v] += 1
        top_edge[o_v] += 1

    return 0

def load_csr_graph(filename):
    """
    Loads graph from a file. Every line is of the format "V_origin,V_destination,Edge_weight".
    Returns a scipy.sparse.csr_matrix with the data.
    """
    raw = np.genfromtxt(filename, delimiter = ",", dtype = np.int32)
    sp_raw = csr_matrix((raw[:,2],(raw[:,0],raw[:,1])))
    del raw

    return sp_raw

def get_boruvka_format(csr_mat):
    """
    Receives a scipy.sparse.csr_matrix with the graph and outputs
    the 4 components necessary for the full representation of the
    graph for the Boruvka algorithm.
    """
    dest = csr_mat.indices
    weight = csr_mat.data
    firstEdge = csr_mat.indptr[:-1]
    outDegree = np.empty_like(firstEdge)

    outdegree_from_firstedge(firstEdge, outDegree, dest.size)

    return dest, weight, firstEdge, outDegree


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

four_elt_mat = np.genfromtxt("datasets/graphs/4elt.edges", delimiter=" ",
                              dtype=[("firstedge","i4"),("dest","i4"),("weight","f4")],
                              skip_header=1)
four_elt_mat_s = csr_matrix((four_elt_mat["weight"], (four_elt_mat["firstedge"], four_elt_mat["dest"])))

del four_elt_mat

four_elt = dict()
four_elt["dest"] = four_elt_mat_s.indices
four_elt["weight"] = four_elt_mat_s.data
four_elt["firstEdge"] = four_elt_mat_s.indptr[:-1]
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

    dest, weight, firstEdge, outDegree = load_graph("4elt")

    t1 = Timer()
    t1.tic()
    mst, n_edges = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
    t1.tac()

    print "mst size", mst.size

    if n_edges < mst.size:
        mst = mst[:n_edges]

    print "time elapsed: ", t1.elapsed
    mst.sort() # mst has to be sorted for comparison with device mst because different threads might be able to write first
    print mst
    print n_edges

def device_boruvka():

    print "CUDA BORUVKA"

    dest, weight, firstEdge, outDegree = load_graph("4elt")

    t1 = Timer()
    t1.tic()
    mst, n_edges = boruvka_minho_gpu(dest, weight, firstEdge, outDegree)
    t1.tac()

    if n_edges < mst.size:
        mst = mst[:n_edges]    

    print "time elapsed: ", t1.elapsed
    mst.sort()
    print mst
    print n_edges

def host_vs_device():
    print "HOST VS DEVICE"

    same_sol = list()

    for r in range(20):
        dest, weight, firstEdge, outDegree = load_graph("4elt")

        t1, t2 = Timer(), Timer()

        t1.tic()
        mst1, n_edges1 = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
        t1.tac()

        if n_edges1 < mst1.size:
            mst1 = mst1[:n_edges1]

        t2.tic()
        mst2, n_edges2 = boruvka_minho_gpu(dest, weight, firstEdge, outDegree, MAX_TPB=256)
        t2.tac()

        if n_edges2 < mst2.size:
            mst2 = mst2[:n_edges2]

        same_sol.append(np.in1d(mst1,mst2).all())

    print "Same solution: ", same_sol

    print "Solution CPU cost: ", weight[mst1].sum()
    print "Solution GPU cost: ", weight[mst2].sum()

    print "Host time elapsed:   ", t1.elapsed
    print "Device time elapsed: ", t2.elapsed


def check_colors():

    print "CUDA BORUVKA"

    dest, weight, firstEdge, outDegree = load_graph("4elt")

    print "# vertices: ", firstEdge.size
    print "# edges:    ", dest.size

    print "Computing MST"

    t1 = Timer()
    t1.tic()
    mst, n_edges = boruvka_minho_gpu(dest, weight, firstEdge, outDegree, MAX_TPB=256)
    t1.tac()

    if n_edges < mst.size:
        mst = mst[:n_edges]
    mst = mst[:-2]
    mst.sort()
    print "time elapsed: ", t1.elapsed
    print "mst size :", mst.size

    print "Generating MST graph"
    nod = np.zeros(outDegree.size, dtype = outDegree.dtype)
    nfe = np.empty(firstEdge.size, dtype = firstEdge.dtype)
    ndest = np.empty(mst.size * 2, dtype = dest.dtype)
    nweight = np.empty(mst.size * 2, dtype = weight.dtype)

    t1.tic()
    get_new_graph(dest, weight, firstEdge, outDegree, mst, nod, nfe, ndest, nweight)
    t1.tac()
     
    print "time elapsed: ", t1.elapsed

    print "Computing labels"
    t1.tic()
    colors = getLabels(ndest, nweight, nfe, nod)
    t1.tac()

    print "time elapsed: ", t1.elapsed
    print "# colors:     ", np.unique(colors).size

def mst_cal():
    sp_cal = load_csr_graph("/home/diogoaos/QCThesis/datasets/graphs/USA-road-d.CAL.csr")
    dest, weight, firstEdge, outDegree = get_boruvka_format(sp_cal)
    del sp_cal

    print "# edges:            ", dest.size
    print "# vertices:         ", firstEdge.size
    print "size of graph (MB): ", (dest.size + weight.size + firstEdge.size + outDegree.size) * 4.0 / 1024 / 1024

    times_cpu = list()
    times_gpu = list()
    equal_mst = list()
    t1, t2 = Timer(), Timer()

    for r in range(10):
    	print "cpu round ", r
        t1.tic()
        mst1, n_edges1 = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
        t1.tac()

        print "finished in ", t1.elapsed

        if n_edges1 < mst1.size:
            mst1 = mst1[:n_edges1]

        print "gpu round ", r

        t2.tic()
        mst2, n_edges2 = boruvka_minho_gpu(dest, weight, firstEdge, outDegree, MAX_TPB=512)
        t2.tac()

        print "finished in ", t2.elapsed
        print ""

        if n_edges2 < mst2.size:
            mst2 = mst2[:n_edges2]

        equal_mst.append(np.in1d(mst1,mst2).all())

        if r > 0:
            times_cpu.append(t1.elapsed)
            times_gpu.append(t2.elapsed)

    print equal_mst
    print "average time cpu: ", np.mean(times_cpu)
    print "average time gpu: ", np.mean(times_gpu)

def mst_and_labeling():
    print "NOTHING HERE"



def main(argv):
    valid_args = [0, 1, 2, 3, 4, 5]
    valid_args = map(str,valid_args)
    if len(argv) <= 1 or argv[1] not in valid_args:
        print "0 : test host boruvka"
        print "1 : test device boruvka"
        print "2 : device vs host boruvka"
        print "3 : test device boruvka"
        print "4 : test device boruvka"
    elif argv[1] == "0":
        host_boruvka()
    elif argv[1] == "1":
        device_boruvka()
    elif argv[1] == "2":
        host_vs_device()        
    elif argv[1] == "3":
        check_colors()           
    elif argv[1] == "4":
        mst_cal()         
    elif argv[1] == "5":
        mst_and_labeling()          
        


if __name__ == "__main__":
    main(sys.argv)