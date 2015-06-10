# -*- coding: utf-8 -*-

"""
Tests for MST algorithms.
"""

import numpy as np
from numba import cuda
from timeit import default_timer as timer

from MyML.graph.mst import *

import sys



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#                     GRAPHS

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

simple_graph = dict()
simple_graph["dest"] = np.array([1, 3, 2, 0, 3, 0, 3, 0, 1, 2, 5, 4, 6, 7, 5, 7, 6, 7], dtype = np.int32)
simple_graph["weight"] = np.array([2, 2, 1, 2, 3, 1, 3, 2, 3, 3, 1, 1, 3, 7, 3, 2, 2, 7], dtype = np.float32)
simple_graph["firstEdge"] = np.array([0, 3, 5, 7, 10, 11, 14, 16], dtype = np.int32)
simple_graph["outDegree"] = np.array([3, 2, 2, 3, 1, 3, 2, 2], dtype = np.int32)


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

def load_graph(name):
	# simple graph of 8 vertices and 9 edges

	graph_names = {"simple_graph" : simple_graph}

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

	dest, weight, firstEdge, outDegree = load_graph("simple_graph")

	t1 = Timer()
	t1.tic()
	mst, n_edges = boruvka_minho_seq(dest, weight, firstEdge, outDegree)
	t1.tac()

	print "time elapsed: ", t1.elapsed
	print mst
	print n_edges

def device_boruvka():

	print "CUDA BORUVKA"

	dest, weight, firstEdge, outDegree = load_graph("simple_graph")

	t1 = Timer()
	t1.tic()
	mst, n_edges = boruvka_minho_gpu(dest, weight, firstEdge, outDegree)
	t1.tac()

	print "time elapsed: ", t1.elapsed
	print mst
	print n_edges




def main(argv):
    valid_args = [0, 1]
    valid_args = map(str,valid_args)
    if len(argv) <= 1 or argv[1] not in valid_args:
        print "0 : test host boruvka"
        print "1 : test device boruvka"

    elif argv[1] == "0":
        host_boruvka()
    elif argv[1] == "1":
        device_boruvka()
        


if __name__ == "__main__":
    main(sys.argv)