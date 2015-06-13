# -*- coding: utf-8 -*-
"""
author: Diogo Silva
notes: Boruvka implementation based on Sousa's "A Generic and Highly Efficient Parallel Variant of Boruvka ’s Algorithm"
       connected components from Boruvka
"""


import numpy as np
from MyML.helper.scan import scan_gpu as ex_prefix_sum_gpu, exprefixsumNumbaSingle as ex_prefix_sum_cpu, exprefixsumNumba as ex_prefix_sum_cpu2
from MyML.graph.mst import findMinEdgeNumba, removeMirroredNumba, initColorsNumba, propagateColorsNumba, buildFlag, countNewEdgesNumba, assignInsertNumba
from numba import jit, cuda, void, boolean, int8, int32, float32


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#
#                      NUMBA CPU 
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def connected_comps_seq(dest_in, weight_in, firstEdge_in, outDegree_in):
    dest = dest_in
    weight = weight_in
    firstEdge = firstEdge_in
    outDegree = outDegree_in

    n_vertices = firstEdge.size
    n_edges = dest.size

    n_components = n_vertices

    # still need edge_id for conflict resolution in find_minedge
    edge_id = np.arange(n_edges, dtype = dest.dtype)
    
    #labels = np.empty(n_vertices, dtype = dest.dtype)
    first_iter = True

    # initialize with name top_edge so we can recycle an array between iterations
    top_edge = np.empty(n_components, dtype = dest.dtype)

    final_converged = False
    while(not final_converged):
        vertex_minedge = top_edge
        findMinEdgeNumba(vertex_minedge, weight, firstEdge, outDegree, edge_id)
        removeMirroredNumba(vertex_minedge, dest)

        # intialize colors of current graph
        colors = np.empty(n_components, dtype = dest.dtype)
        initColorsNumba(vertex_minedge, dest, colors)

        # propagate colors until convergence
        converged = False
        while(not converged):
            converged = propagateColorsNumba(colors)

        # flag marks the vertices that are the representatives of the new supervertices
        # new_vertex will be initialized with he flags
        new_vertex = vertex_minedge
        buildFlag(colors, new_vertex)
    
        new_n_vertices = ex_prefix_sum_cpu(new_vertex, init = 0)

        if first_iter:
            # first iteration defines labels as the initial colors and updates
            labels = colors.copy()
            update_labels_single_pass(labels, colors, new_vertex)
            first_iter = False
        else:
            # other iterations update the labels with the new colors
            update_labels_single_pass(labels, colors, new_vertex)        


        if new_n_vertices == 1:
            final_converged = True
            break        

        # count number of edges for new supervertices and write in new outDegree
        newOutDegree = np.zeros(new_n_vertices, dtype = dest.dtype)
        countNewEdgesNumba(colors, firstEdge, outDegree, dest, new_vertex, newOutDegree)

        # new first edge array for contracted graph
        newFirstEdge = np.empty(newOutDegree.size, dtype = dest.dtype)
        new_n_edges = ex_prefix_sum_cpu2(newOutDegree, newFirstEdge, init = 0)

        # if no edges remain, then MST has converged
        if new_n_edges == 0:
            final_converged = True
            break

        # create arrays for new edges
        new_dest = np.empty(new_n_edges, dtype = dest.dtype)
        new_edge_id = np.empty(new_n_edges, dtype = dest.dtype)
        new_weight = np.empty(new_n_edges, dtype = weight.dtype)
        top_edge = newFirstEdge.copy()

        # assign and insert new edges
        assignInsertNumba(edge_id, dest, weight, firstEdge, outDegree, colors,
                          new_vertex, new_dest, new_edge_id, new_weight, top_edge)

        # delete old graph
        del new_vertex, edge_id, dest, weight, firstEdge, outDegree, colors

        # write new graph
        n_components = newFirstEdge.size
        edge_id = new_edge_id
        dest = new_dest
        weight = new_weight
        firstEdge = newFirstEdge
        outDegree = newOutDegree


    return labels


@jit
def update_labels_numba(labels, update_array):
    """
    This kernel is dual purpose.

    This kernel updates the color of each vertex in the graph.
    The current colors of the vertices are indices for the current components 
    in the graph. Each current component in the graph will have a new color.
    The new color of vertex v will be the new color colors[curr_color] of its
    color labels[v] (index of a component). This step happens because the new
    color propagation is representative of the contracted graph. In this phase
    the update_array is the colors array.

    Old components are merged into new ones with new IDs. The labels need to
    have the new IDs. In this phase the update_array is the new_vertex array.
    """
    n_components = labels.size

    for v in range(n_components):
        curr_color = labels[v]
        labels[v] = update_array[curr_color]

@jit
def update_labels_single_pass(labels, colors, new_vertex):
    """
    Does all the updates on a single pass
    """
    n_components = labels.size

    for v in range(n_components):
        curr_color = labels[v]
        new_color = colors[curr_color]
        new_color_id = new_vertex[new_color]
        labels[v] = new_color_id

@cuda.jit
def update_labels_cuda(labels, update_array):
    """
    CUDA version of update_labels.
    """

    v = cuda.grid(1)
    n_components = labels.size

    if v >= n_components:
        return

    curr_color = labels[v]
    labels[v] = update_array[curr_color]

@jit
def update_labels_single_pass_cuda(labels, colors, new_vertex):
    """
    Does all the updates on a single pass
    """
    v = cuda.grid(1)
    n_components = labels.size

    if v >= n_components:
        return    

    curr_color = labels[v]
    new_color = colors[curr_color]
    new_color_id = new_vertex[new_color]
    labels[v] = new_color_id