import numpy as np

import numbapro
from numbapro import cuda

# Sequential Prefix Sum
def prefixsum(masks, indices, init=0, nelem=None):
    nelem = masks.size if nelem is None else nelem

    carry = init
    for i in range(nelem):
        indices[i] = carry
        if masks[i]:
            carry += 1

    indices[nelem] = carry
    return carry

# Parallel CUDA Prefix Sum
@cuda.autojit
def cuda_prefixsum_base2(masks, indices, init, nelem):
    """
    Args
    ----
    nelem:
        Must be power of 2.
    Note
    ----
    Launch 2*nelem threads.  Support 1 block/grid.
    """
    sm = cuda.shared.array((1024,), dtype=numba.int64)
    tid = cuda.threadIdx.x

    # Preload
    if 2 * tid + 1 < nelem:
        sm[2 * tid] = masks[2 * tid]
        sm[2 * tid + 1] = masks[2 * tid + 1]

    # Up phase
    limit = nelem >> 1
    step = 1
    idx = tid * 2
    two_d = 1
    for d in range(3):
        offset = two_d - 1

        if tid < limit:
            sm[offset + idx + step] += sm[offset + idx]

        limit >>= 1
        idx <<= 1
        step <<= 1
        two_d <<= 1

    cuda.syncthreads()

    # Down phase

    if tid == 0:
        sm[nelem - 1] = 0


    cuda.syncthreads()

    # Writeback
    if 2 * tid + 1 < nelem:
        indices[2 * tid] = sm[2 * tid]
        indices[2 * tid + 1] = sm[2 * tid + 1]


class GraphCSR():
    def __init__(self):
        pass

    def buildFromKNN(self,values,neighbours):
        n_samples, n_neigh = values.shape

        self.n_edges = values.nonzero()[0].size





class BoruvkaMinhoGPU():


    def __init__(self):
        pass

    def fit(self,graph):
        """
        graph         : graph in the CSR format:
                        destination    - an array of length E with the destination of each edges
                        weight         - an array of length E with the weight of each edge
                        firstedge     - an array of length V with the ID of the first edge of each vertex
                        outdegree     - an array of length V with the number of outgoing edges of each vertex
        """

        self.n_edges = graph.dest.size
        self.n_vertices = graph.minedge.size

        self.n_e_current = self.n_edges # current number of edges
        self.n_v_current = self.n_vertices #current number of vertices




    def findMinEdge(self,graph):
        vertex_minedge = np.empty(self.n_v_current)


    
    @numbapro.cuda.jit("void(float32[:], int32[:], int32[:], int32[:])")
    def findMinEdge_kernel(weight,firstedge,outdegree,vertex_minedge):
        # thread ID inside block
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # block ID
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y

        # block dimensions
        bw = cuda.blockDim.x
        bh = cuda.blockDim.y

        # grid dimensions
        gw = cuda.gridDim.x
        gh = cuda.gridDim.y

        # compute thread's x and y index (i.e. datapoint and cluster)
        # tx doesn't matter
        # the second column of blocks means we want to add
        # 2**16 to the index
        n = ty + by * bh + bx*gh*bh


        # a thread per vertex
        if n >= firstedge.size:
            return

        ########################
        # n : the index of the vertex to compute
        start = firstedge[n] # initial edge
        end = start + outdegree[n] # initial edge of next vertex

        min_edge = weight[start] # get first weight for comparison inside loop

        # loop through all the edges of vertex to get the minimum
        for i in range(start+1,end):
            temp = weight[i]
            if temp < min_edge:
                min_edge = temp

        vertex_minedge[n] = min_edge
    
    @numbapro.cuda.jit("void(int32[:], int32[:],int32[:])")
    def removeMirroredEdges(destination,vertex_minedge,colours):
        # thread ID inside block
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # block ID
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y

        # block dimensions
        bw = cuda.blockDim.x
        bh = cuda.blockDim.y

        # grid dimensions
        gw = cuda.gridDim.x
        gh = cuda.gridDim.y

        # compute thread's x and y index (i.e. datapoint and cluster)
        # tx doesn't matter
        # the second column of blocks means we want to add
        # 2**16 to the index
        n = ty + by * bh + bx*gh*bh

        # a thread per vertex
        if n >= vertex_minedge.size:
            return

        ########################
        my_edge = vertex_minedge[n]

        if my_edge == -1:
            return

        my_successor = destination[my_edge]
        successor_edge = vertex_minedge[my_successor]

        # successor already processed and its edge removed
        # because it was a mirrored edge with this vertex or another
        # either way nothing to do here
        if successor_edge == -1:
            return

        successor_successor = destination[successor_edge]

        # if the successor of the vertex's successor is the vertex itself AND
        # the vertex's ID is bigger than its successor, than remove its edge
        if n == successor_successor:
            if n < my_successor:
                vertex_minedge[n] = -1
            else:
                vertex_minedge[my_successor] = -1



    @numbapro.cuda.jit("void(int32[:], int32[:],int32[:])")
    def initializeColours(destination,vertex_minedge,colours):
        # thread ID inside block
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # block ID
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y

        # block dimensions
        bw = cuda.blockDim.x
        bh = cuda.blockDim.y

        # grid dimensions
        gw = cuda.gridDim.x
        gh = cuda.gridDim.y

        # compute thread's x and y index (i.e. datapoint and cluster)
        # tx doesn't matter
        # the second column of blocks means we want to add
        # 2**16 to the index
        n = ty + by * bh + bx*gh*bh

        # a thread per vertex
        if n >= vertex_minedge.size:
            return

        ########################
        my_edge = vertex_minedge[n]

        if my_edge == -1:
            colours[n] = n
        else:
            my_successor = destination[my_edge]
            colours[n] = my_successor


    @numbapro.cuda.jit("void(int32[:])")
    def propagateColours(colours):
        # thread ID inside block
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        # block ID
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y

        # block dimensions
        bw = cuda.blockDim.x
        bh = cuda.blockDim.y

        # grid dimensions
        gw = cuda.gridDim.x
        gh = cuda.gridDim.y

        # compute thread's x and y index (i.e. datapoint and cluster)
        # tx doesn't matter
        # the second column of blocks means we want to add
        # 2**16 to the index
        n = ty + by * bh + bx*gh*bh

        # a thread per vertex
        if n >= colours.shape[0]:
            return

        ########################
        my_colour = colours[n] # colour of vertex # n
        colour_of_successor = colours[my_colour] # colour of successor of vertex

        # if my colour is different from that of my successor
        if my_colour != colour_of_successor:
            colours[n]=colour_of_successor

        






def minimum_spanning_tree(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()
 
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
    
    # set to keep track of unvisited vertices
    # useful for unconnected graphs
    unvisited_vertices=set(xrange(n_vertices)) - {0}
    
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
    
    start_of_mst = [0]
     
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)

        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
        
        # when min is inf, it means the graph is unconnected
        # we add a vertex from the unvisited set
        if X[new_edge[0],new_edge[1]] == np.inf:
            added_vertex = unvisited_vertices.pop()
            new_edge=[added_vertex,np.argmin(X[added_vertex])]
            visited_vertices.append(added_vertex) # add poped vertex to visited
            num_visited += 1
            
            start_of_mst.append(num_visited) # add start of new independent MST
        
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        
        # remove vertex from unvisited
        unvisited_vertices.discard(new_edge[1])
        
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
    return np.vstack(spanning_edges),np.array(start_of_mst)


def minimum_spanning_tree_csr(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()
 
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")

    n_vertices = X.shape[0]
    spanning_edges = []
    
    # set to keep track of unvisited vertices
    # useful for unconnected graphs
    unvisited_vertices = set(xrange(n_vertices)) - {0}
    
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    #diag_indices = np.arange(n_vertices)
    #X[diag_indices, diag_indices] = np.inf
    
    start_of_mst = [0]
    mst_edges = set()

    while num_visited != n_vertices:
        # get shortest edge from visited vertices

        min_weight = np.inf


        for vertex in visited_vertices:

            # check if vertex has any edges
            np.where(X[vertex].data < min_weight)



        new_edge = np.argmin(X[visited_vertices], axis=None)

        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
        
        # when min is inf, it means the graph is unconnected
        # we add a vertex from the unvisited set
        if X[new_edge[0],new_edge[1]] == np.inf:
            added_vertex = unvisited_vertices.pop()
            new_edge=[added_vertex,np.argmin(X[added_vertex])]
            visited_vertices.append(added_vertex) # add poped vertex to visited
            num_visited += 1
            
            start_of_mst.append(num_visited) # add start of new independent MST
        
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        
        # remove vertex from unvisited
        unvisited_vertices.discard(new_edge[1])
        
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
    return np.vstack(spanning_edges),np.array(start_of_mst)


class boruvkaMinhoSeq():
    
    def __init__(self, dest, weight, firstEdge, outDegree):
        self.dest = dest
        self.weight = weight
        self.firstEdge = firstEdge
        self.outDegree = outDegree

        
        self.n_vertices = firstEdge.size
        self.n_edges = dest.size

        self.edge_id = np.arange(self.n_edges)
        
        # total edges is (|V| - 1) * 2 because edges are duplicated to cover each direction
        
        
        self.n_components = self.n_vertices
        self.n_mst = 1
        
        
    def fit(self):
        self.mst = np.empty(self.n_vertices - 1, dtype = np.int32)
        self.mst_pointer = 0

        while(self.n_components > 1):

            self.findMinEdge()
            self.removeMirrored()
            
            edge_args = self.vertex_minedge[self.vertex_minedge != -1] # args of unremoved edges
            self.mst[self.mst_pointer:self.mst_pointer + edge_args.size] = self.edge_id[edge_args] # edge IDs
            self.mst_pointer += edge_args.size # increment pointer

            self.initColors()

            while (not self.converged):
                self.propagateColors()

               # if all colors are the same, stop here
               # if (self.colors == self.colors[0]).all():
               #     self.n_components = 1
               #     break

            self.createNewVertexID()
            del self.vertex_minedge

            self.countNewVertex()

            self.assignInsert()

        return self.mst

            
    def findMinEdge(self):
        vertex_minedge = np.empty(self.n_components, dtype = np.int32)

        for v in xrange(self.n_components):
            if self.outDegree[v] == 0:
                raise Exception("Graph not fully connected.")
            startW = self.firstEdge[v]
            endW = startW + self.outDegree[v]
            # we're slicing the array so the result of argmin is offset by startW
            edge_arg= startW + np.argmin(self.weight[startW:endW])
            vertex_minedge[v] = edge_arg #self.edge_id[edge_arg]
        self.vertex_minedge = vertex_minedge
    
    def removeMirrored(self):
        for v in xrange(self.n_components): # for each vertex
            
            myEdge = self.vertex_minedge[v] # my edge
            my_succ = self.dest[myEdge] # my successor
            succ_edge = self.vertex_minedge[my_succ] # my successor's edge
            
            # if my successor's edge is -1 it means it was already removed
            if succ_edge == -1:
                continue
            
            succ_succ = self.dest[succ_edge] # my successor's successor
            
            # if my successor's successor is me then remove the edge of either me or him
            # depending on which of us has a lower ID
            if v == succ_succ:
                self.vertex_minedge[min(v,my_succ)] = -1
        
    def initColors(self):
        colors = np.empty(self.n_components, dtype = np.int32)
        self.converged = False
        
        for v in xrange(self.n_components):
            my_edge = self.vertex_minedge[v]
            if my_edge != -1:
                colors[v] = self.dest[my_edge]
            else:
                colors[v] = v
        
        self.colors = colors

    @numbapro.autojit
    def initColorsNumba(self, n_components, vertex_minedge, dest, colors):
        #colors = np.empty(n_components, dtype = np.int32)
        
        for v in xrange(n_components):
            my_edge = vertex_minedge[v]
            if my_edge != -1:
                colors[v] = dest[my_edge]
            else:
                colors[v] = v
        
        return colors
        
    def propagateColors(self):
        """
        For checking convergence, start with a boolean variable converged set to True.
        At each assignment the new color is compared to the old one. The result
        of this comparison (True of False) is used to perform a boolean AND with converged.
        If all the new colors are equal to the old colors then the result of all the ANDs is
        True and convergence was met.
        """
        converged = True
        for v in xrange(self.n_components): # for each vertex
            my_color = self.colors[v] # my_color is also my successor
            if my_color != v:
                new_color = self.colors[my_color] # my new color is the color of my successor
                converged = converged and (new_color == my_color) # check if color changed
                self.colors[v] = new_color # assign new color
        self.converged = converged

    @numbapro.autojit
    def propagateColorsNumba(self, n_components, colors):
        """
        For checking convergence, start with a boolean variable converged set to True.
        At each assignment the new color is compared to the old one. The result
        of this comparison (True of False) is used to perform a boolean AND with converged.
        If all the new colors are equal to the old colors then the result of all the ANDs is
        True and convergence was met.
        """
        converged = 0
        for v in xrange(n_components): # for each vertex
            my_color = colors[v] # my_color is also my successor
            if my_color != v:
                new_color = colors[my_color] # my new color is the color of my successor
                if new_color != my_color:
                    converged = 1
                colors[v] = new_color # assign new color
        return converged            
    
    def createNewVertexID(self):
        flag = self.vertex_minedge == -1 # get super-vertives representatives
        indices = np.empty(self.n_components, dtype = np.int32) # new indices
        
        exprefixsum(flag, indices, init = 0, nelem = self.n_components)
        
        self.new_vertex = indices
        
    def countNewVertex(self):
        # new number of vertices is the number of representatives
        outDegree = np.zeros(self.new_vertex[-1], dtype = np.int32)
        
        for v in xrange(self.n_components):
            my_color = self.colors[v] # my color
            my_color_id = self.new_vertex[my_color] # vertex id of my color

            # my edges
            startW = self.firstEdge[v]
            endW = startW + self.outDegree[v]
            
            my_succs = self.dest[startW:endW] # my successors
            my_succ_colors = self.colors[my_succs] # my successors' colors
            n_alien_edges = (my_succ_colors != my_color).sum()
            
            #alien_edges = np.where(my_succ_colors != my_color)[0] # arg of edges that lead to succ whose color is different
            # where gives an offset arg, need to add startW to correct it
            #alien_edges = startW + alien_edges
            
            outDegree[my_color_id] += n_alien_edges # increment number of outgoing edges of super-vertex
            
        new_first_edge = np.empty_like(outDegree)
        exprefixsum(outDegree, new_first_edge, init = 0, nelem = None)
        
        self.new_first_edge = new_first_edge
        self.new_outDegree = outDegree


        
    def assignInsert(self):
        
        top_edge = self.new_first_edge.copy() # pointer to new destination indices
        next_num_edges = self.new_outDegree.sum() # number of edges in new contracted graph
        new_dest = np.empty(next_num_edges, dtype = np.int32) # new destination array
        new_edge_id = np.empty(next_num_edges, dtype = np.int32) # new edge id array
        new_weight = np.empty(next_num_edges, dtype = np.float32) # new weight array
        
        for v in xrange(self.n_components):
            my_color = self.colors[v] # my color
            

            # my edges
            startW = self.firstEdge[v] 
            endW = startW + self.outDegree[v]

            for edge in xrange(startW,endW):
                my_succ = self.dest[edge] # my successor
                my_succ_color = self.colors[my_succ] # my successor's color

                # keep edge if colors are different
                if my_color != my_succ_color:
                    supervertex_id = self.new_vertex[my_color]
                    succ_supervertex_id = self.new_vertex[my_succ_color] # supervertex id of my succ color

                    pointer = top_edge[supervertex_id] # where to add edge
                    
                    new_dest[pointer] = succ_supervertex_id
                    new_weight[pointer] = self.weight[edge]
                    new_edge_id[pointer] = self.edge_id[edge]

                    top_edge[supervertex_id] += 1 # increment pointer of current supervertex
        
        # number of components of new contracted graph
        self.n_components = self.new_first_edge.size

        # update arrays describing the graph
        self.dest = new_dest
        self.weight = new_weight
        self.edge_id = new_edge_id
        self.firstEdge = self.new_first_edge
        self.outDegree = self.new_outDegree

        del new_dest, new_weight, new_edge_id, self.new_first_edge, self.new_outDegree, self.colors


class boruvkaMinhoSeqForest():
    
    def __init__(self, dest, weight, firstEdge, outDegree):
        self.dest = dest
        self.weight = weight
        self.firstEdge = firstEdge
        self.outDegree = outDegree

        
        self.n_vertices = firstEdge.size
        self.n_edges = dest.size

        self.edge_id = np.arange(self.n_edges, dtype = np.int32)
        
        # total edges is (|V| - 1) * 2 because edges are duplicated to cover each direction
        
        
        self.n_components = self.n_vertices
        self.n_mst = 1

        self.final_converged = False
        
        
    def fit(self):
        """
        Returns an array of length (# vertices - 1). Each element of this array points
        to an edge in the original graph CSR representation.

        In case the input graph is unconnected than there are multiple MST in the result.
        There will be an element == -1 for each independent MST inside the graph. All the
        -1 elements will be at the end. To know how many MST the graph has, one has only to
        count the number of -1 elements, starting from the end to check less elements.
        """
        # maximum size of MST is when it is connected
        self.mst = np.empty(self.n_vertices - 1, dtype = np.int32)
        # if there are elements == -1 in the end it means 
        # there
        self.mst.fill(-1)
        self.mst_pointer = 0

        #while(self.n_components > self.n_mst):
        while(not self.final_converged):

            self.findMinEdge()
            self.removeMirrored()
            
            self.addEdgesToMST()

            self.initColors()

            while (not self.converged):
                self.propagateColors()

               # if all colors are the same, stop here
               # if (self.colors == self.colors[0]).all():
               #     self.n_components = 1
               #     break

            self.createNewVertexID()
            del self.vertex_minedge

            self.countNewVertex()

            self.assignInsert()

            self.checkConvergence()

        return self.mst

    def fitNumba(self):
        # maximum size of MST is when it is connected
        self.mst = np.empty(self.n_vertices - 1, dtype = np.int32)
        # if there are elements == -1 in the end it means 
        # there
        self.mst.fill(-1)
        self.mst_pointer = 0

        while(not self.final_converged):
            vertex_minedge = np.empty(self.n_components, dtype = np.int32)
            findMinEdgeNumba(vertex_minedge, self.weight, self.firstEdge, self.outDegree)
            removeMirroredNumba(vertex_minedge, self.dest)

            # add new edges to final MST and update MST pointer
            self.mst_pointer = addEdgesToMSTNumba(self.mst, self.mst_pointer, vertex_minedge, self.edge_id)

            # intialize colors of current graph
            colors = np.empty(self.n_components, dtype = np.int32)
            initColorsNumba(vertex_minedge, self.dest, colors)

            # propagate colors until convergence
            converged = False
            while(not converged):
                converged = propagateColorsNumba(colors)

            # flag marks the vertices that are the representatives of the new supervertices
            flag = np.where(vertex_minedge == -1, 1, 0).astype(np.int32) # get super-vertives representatives
            del vertex_minedge # vertex_minedge no longer necessary for next steps

            # new supervertices indices
            new_vertex = np.empty(self.n_components, dtype = np.int32) # new indices       
            exprefixsumNumba(flag, new_vertex, init = 0)

            del flag # no longer need flag

            # count number of edges for new supervertices and write in new outDegree
            newOutDegree = np.zeros(new_vertex[-1], dtype = np.int32)
            countNewEdgesNumba(colors, self.firstEdge, self.outDegree, self.dest, new_vertex, newOutDegree)

            # new first edge array for contracted graph
            newFirstEdge = np.empty(newOutDegree.size, dtype = np.int32)
            totalEdges = exprefixsumNumba(newOutDegree, newFirstEdge, init = 0)

            # if no edges remain, then MST has converged
            if totalEdges == 0:
                self.final_converged = True
                break

            # create arrays for new edges
            new_dest = np.empty(totalEdges, dtype = np.int32)
            new_edge_id = np.empty(totalEdges, dtype = np.int32)
            new_weight = np.empty(totalEdges, dtype = np.float32)
            top_edge = newFirstEdge.copy()

            # assign and insert new edges
            assignInsertNumba(self.edge_id, self.dest, self.weight, self.firstEdge, self.outDegree, colors, new_vertex, newFirstEdge, new_dest, new_edge_id, new_weight, top_edge)

            # delete old graph
            del new_vertex, self.edge_id, self.dest, self.weight, self.firstEdge, self.outDegree, colors, top_edge

            # write new graph
            self.n_components = newFirstEdge.size
            self.edge_id = new_edge_id
            self.dest = new_dest
            self.weight = new_weight
            self.firstEdge = newFirstEdge
            self.outDegree = newOutDegree


        return self.mst




    def addEdgesToMST(self):
        # args of unremoved edges
        edge_args = self.vertex_minedge[self.vertex_minedge != -1]
        # edge IDs
        self.mst[self.mst_pointer:self.mst_pointer + edge_args.size] = self.edge_id[edge_args]
        # increment pointer
        self.mst_pointer += edge_args.size
            
    def findMinEdge(self):
        vertex_minedge = np.empty(self.n_components, dtype = np.int32)

        for v in xrange(self.n_components):
            if self.outDegree[v] == 0:
                vertex_minedge[v] = -1
                continue
            startW = self.firstEdge[v]
            endW = startW + self.outDegree[v]
            # we're slicing the array so the result of argmin is offset by startW
            edge_arg= startW + np.argmin(self.weight[startW:endW])
            vertex_minedge[v] = edge_arg #self.edge_id[edge_arg]
        self.vertex_minedge = vertex_minedge


    
    def removeMirrored(self):
        for v in xrange(self.n_components): # for each vertex
            
            myEdge = self.vertex_minedge[v] # my edge

            # if my edge is -1 it means it was already removed
            if myEdge == -1:
                continue

            my_succ = self.dest[myEdge] # my successor
            succ_edge = self.vertex_minedge[my_succ] # my successor's edge
            
            # if my successor's edge is -1 it means it was already removed
            if succ_edge == -1:
                continue
            
            succ_succ = self.dest[succ_edge] # my successor's successor
            
            # if my successor's successor is me then remove the edge of either me or him
            # depending on which of us has a lower ID
            if v == succ_succ:
                self.vertex_minedge[min(v,my_succ)] = -1

    def initColors(self):
        colors = np.empty(self.n_components, dtype = np.int32)
        self.converged = False
        
        for v in xrange(self.n_components):
            my_edge = self.vertex_minedge[v]
            if my_edge != -1:
                colors[v] = self.dest[my_edge]
            else:
                colors[v] = v
        
        self.colors = colors
        
    def propagateColors(self):
        """
        For checking convergence, start with a boolean variable converged set to True.
        At each assignment the new color is compared to the old one. The result
        of this comparison (True of False) is used to perform a boolean AND with converged.
        If all the new colors are equal to the old colors then the result of all the ANDs is
        True and convergence was met.
        """
        converged = True
        for v in xrange(self.n_components): # for each vertex
            my_color = self.colors[v] # my_color is also my successor
            if my_color != v:
                new_color = self.colors[my_color] # my new color is the color of my successor
                converged = converged and (new_color == my_color) # check if color changed
                self.colors[v] = new_color # assign new color
        self.converged = converged
    
    def createNewVertexID(self):
        """
        TODO:
            self.vertex_minedge can be reutilized as the new_vertex array
            since it is not needed from here on
        """
        flag = self.vertex_minedge == -1 # get super-vertives representatives
        

        indices = np.empty(self.n_components, dtype = np.int32) # new indices
        
        exprefixsum(flag, indices, init = 0, nelem = self.n_components)
        
        self.new_vertex = indices
        
    def countNewVertex(self):
        # new number of vertices is the number of representatives
        outDegree = np.zeros(self.new_vertex[-1], dtype = np.int32)
        
        for v in xrange(self.n_components):
            my_color = self.colors[v] # my color
            my_color_id = self.new_vertex[my_color] # vertex id of my color

            # my edges
            startW = self.firstEdge[v]
            endW = startW + self.outDegree[v]
            
            my_succs = self.dest[startW:endW] # my successors
            my_succ_colors = self.colors[my_succs] # my successors' colors
            n_alien_edges = (my_succ_colors != my_color).sum()
            
            #alien_edges = np.where(my_succ_colors != my_color)[0] # arg of edges that lead to succ whose color is different
            # where gives an offset arg, need to add startW to correct it
            #alien_edges = startW + alien_edges
            
            outDegree[my_color_id] += n_alien_edges # increment number of outgoing edges of super-vertex
            
        new_first_edge = np.empty_like(outDegree)
        exprefixsum(outDegree, new_first_edge, init = 0, nelem = None)
        
        self.new_first_edge = new_first_edge
        self.new_outDegree = outDegree



        
    def assignInsert(self):
        
        top_edge = self.new_first_edge.copy() # pointer to new destination indices
        next_num_edges = self.new_outDegree.sum() # number of edges in new contracted graph
        new_dest = np.empty(next_num_edges, dtype = np.int32) # new destination array
        new_edge_id = np.empty(next_num_edges, dtype = np.int32) # new edge id array
        new_weight = np.empty(next_num_edges, dtype = np.float32) # new weight array
        
        for v in xrange(self.n_components):
            my_color = self.colors[v] # my color
            

            # my edges
            startW = self.firstEdge[v] 
            endW = startW + self.outDegree[v]

            for edge in xrange(startW,endW):
                my_succ = self.dest[edge] # my successor
                my_succ_color = self.colors[my_succ] # my successor's color

                # keep edge if colors are different
                if my_color != my_succ_color:
                    supervertex_id = self.new_vertex[my_color]
                    succ_supervertex_id = self.new_vertex[my_succ_color] # supervertex id of my succ color

                    pointer = top_edge[supervertex_id] # where to add edge
                    
                    new_dest[pointer] = succ_supervertex_id
                    new_weight[pointer] = self.weight[edge]
                    new_edge_id[pointer] = self.edge_id[edge]

                    top_edge[supervertex_id] += 1 # increment pointer of current supervertex
        
        # number of components of new contracted graph
        self.n_components = self.new_first_edge.size

        # update arrays describing the graph
        self.dest = new_dest
        self.weight = new_weight
        self.edge_id = new_edge_id
        self.firstEdge = self.new_first_edge
        self.outDegree = self.new_outDegree

        del new_dest, new_weight, new_edge_id, self.new_first_edge, self.new_outDegree, self.colors


    def checkConvergence(self):
        """
        If there are no outgoing edges, MST forest converged.
        """

        if not self.outDegree.any():
            self.final_converged = True

def boruvka_minho_seq(dest, weight, firstEdge, outDegree):
    dest = dest
    weight = weight
    firstEdge = firstEdge
    outDegree = outDegree

    
    n_vertices = firstEdge.size
    n_edges = dest.size

    edge_id = np.arange(n_edges, dtype = np.int32)
    
    # total edges is (|V| - 1) * 2 because edges are duplicated to cover each direction
    
    
    n_components = n_vertices
    n_mst = 1

    final_converged = False
    # maximum size of MST is when it is connected
    mst = np.empty(n_vertices - 1, dtype = np.int32)
    # if there are elements == -1 in the end it means 
    # there
    mst.fill(-1)
    mst_pointer = 0

    while(not final_converged):
        vertex_minedge = np.empty(n_components, dtype = np.int32)
        findMinEdgeNumba(vertex_minedge, weight, firstEdge, outDegree)
        removeMirroredNumba(vertex_minedge, dest)

        # add new edges to final MST and update MST pointer
        mst_pointer = addEdgesToMSTNumba(mst, mst_pointer, vertex_minedge, edge_id)

        # intialize colors of current graph
        colors = np.empty(n_components, dtype = np.int32)
        initColorsNumba(vertex_minedge, dest, colors)

        # propagate colors until convergence
        converged = False
        while(not converged):
            converged = propagateColorsNumba(colors)

        # flag marks the vertices that are the representatives of the new supervertices
        flag = np.where(vertex_minedge == -1, 1, 0).astype(np.int32) # get super-vertives representatives
        del vertex_minedge # vertex_minedge no longer necessary for next steps

        # new supervertices indices
        new_vertex = np.empty(n_components, dtype = np.int32) # new indices       
        exprefixsumNumba(flag, new_vertex, init = 0)

        del flag # no longer need flag

        # count number of edges for new supervertices and write in new outDegree
        newOutDegree = np.zeros(new_vertex[-1], dtype = np.int32)
        countNewEdgesNumba(colors, firstEdge, outDegree, dest, new_vertex, newOutDegree)

        # new first edge array for contracted graph
        newFirstEdge = np.empty(newOutDegree.size, dtype = np.int32)
        totalEdges = exprefixsumNumba(newOutDegree, newFirstEdge, init = 0)

        # if no edges remain, then MST has converged
        if totalEdges == 0:
            final_converged = True
            break

        # create arrays for new edges
        new_dest = np.empty(totalEdges, dtype = np.int32)
        new_edge_id = np.empty(totalEdges, dtype = np.int32)
        new_weight = np.empty(totalEdges, dtype = np.float32)
        top_edge = newFirstEdge.copy()

        # assign and insert new edges
        assignInsertNumba(edge_id, dest, weight, firstEdge, outDegree, colors, new_vertex, newFirstEdge, new_dest, new_edge_id, new_weight, top_edge)

        # delete old graph
        del new_vertex, edge_id, dest, weight, firstEdge, outDegree, colors, top_edge

        # write new graph
        n_components = newFirstEdge.size
        edge_id = new_edge_id
        dest = new_dest
        weight = new_weight
        firstEdge = newFirstEdge
        outDegree = newOutDegree


    return mst

@numbapro.jit(numbapro.void(numbapro.int32[:],numbapro.float32[:],numbapro.int32[:],numbapro.int32[:]))
def findMinEdgeNumba(vertex_minedge, weight, firstEdge, outDegree):

    n_components = vertex_minedge.size

    for v in range(n_components):
        if outDegree[v] == 0:
            vertex_minedge[v] = -1
            continue
        startW = firstEdge[v]
        endW = startW + outDegree[v]
        # we're slicing the array so the result of argmin is offset by startW
        edge_arg = startW + np.argmin(weight[startW:endW])
        vertex_minedge[v] = edge_arg #self.edge_id[edge_arg]

@numbapro.jit(numbapro.void(numbapro.int32[:],numbapro.int32[:]))
def removeMirroredNumba(vertex_minedge, dest):
    n_components = vertex_minedge.size
    for v in range(n_components): # for each vertex
        
        myEdge = vertex_minedge[v] # my edge

        if myEdge == -1:
            continue

        my_succ = dest[myEdge] # my successor
        succ_edge = vertex_minedge[my_succ] # my successor's edge
        
        # if my successor's edge is -1 it means it was already removed
        if succ_edge == -1:
            continue
        
        succ_succ = dest[succ_edge] # my successor's successor
        
        # if my successor's successor is me then remove the edge of either me or him
        # depending on which of us has a lower ID
        if v == succ_succ:
            if v < my_succ:
                vertex_minedge[v] = -1
            else:
                vertex_minedge[my_succ] = -1

@numbapro.jit(numbapro.int32(numbapro.int32[:],numbapro.int32,numbapro.int32[:],numbapro.int32[:]))
def addEdgesToMSTNumba(mst, mst_pointer, vertex_minedge, edge_id):
        n_components = vertex_minedge.size

        for v in range(n_components):
            my_edge = vertex_minedge[v]
            if my_edge != -1:
                mst[mst_pointer] = edge_id[my_edge]
                mst_pointer += 1
        return mst_pointer

@numbapro.jit(numbapro.void(numbapro.int32[:],numbapro.int32[:],numbapro.int32[:]))
def initColorsNumba(vertex_minedge, dest, colors):

        n_components = vertex_minedge.size
        
        for v in range(n_components):
            my_edge = vertex_minedge[v]
            if my_edge != -1:
                colors[v] = dest[my_edge]
            else:
                colors[v] = v

@numbapro.jit(numbapro.boolean(numbapro.int32[:]))
def propagateColorsNumba(colors):
    """
    For checking convergence, start with a boolean variable converged set to True.
    At each assignment the new color is compared to the old one. The result
    of this comparison (True of False) is used to perform a boolean AND with converged.
    If all the new colors are equal to the old colors then the result of all the ANDs is
    True and convergence was met.
    """
    n_components = colors.size
    converged = True
    for v in range(n_components): # for each vertex
        my_color = colors[v] # my_color is also my successor
        if my_color != v:
            new_color = colors[my_color] # my new color is the color of my successor
            if new_color != my_color:
                converged = False
            colors[v] = new_color # assign new color
    return converged

@numbapro.jit(numbapro.void(numbapro.int32[:],numbapro.int32[:],numbapro.int32[:],numbapro.int32[:],numbapro.int32[:],numbapro.int32[:]))
def countNewEdgesNumba(colors, firstEdge, outDegree, dest, new_vertex, newOutDegree):
    # new number of vertices is the number of representatives
    
    n_components = colors.size
    for v in range(n_components):
        my_color = colors[v] # my color
        my_color_id = new_vertex[my_color] # vertex id of my color (supervertex)

        # my edges
        startW = firstEdge[v]
        endW = startW + outDegree[v]

        for edge in range(startW,endW):
        	my_succ = dest[edge]
        	my_succ_color = colors[my_succ]

        	if my_color != my_succ_color:
        		newOutDegree[my_color_id] += 1 # increment number of outgoing edges of super-vertex

        
        # my_succs = dest[startW:endW] # my successors
        # my_succ_colors = colors[my_succs] # my successors' colors
        # n_alien_edges = (my_succ_colors != my_color).sum()

        # newOutDegree[my_color_id] += n_alien_edges # increment number of outgoing edges of super-vertex
        
@numbapro.jit(numbapro.void(numbapro.int32[:],numbapro.int32[:],numbapro.float32[:],numbapro.int32[:],
			  				numbapro.int32[:],numbapro.int32[:],numbapro.int32[:],numbapro.int32[:],
			  				numbapro.int32[:],numbapro.int32[:],numbapro.float32[:],numbapro.float32[:]))
def assignInsertNumba(edge_id, dest, weight, firstEdge,
					  outDegree, colors, new_vertex, new_first_edge,
					  new_dest, new_edge_id, new_weight, top_edge):

    n_components = colors.size
    
    for v in range(n_components):
        my_color = colors[v] # my color

        # my edges
        startW = firstEdge[v] 
        endW = startW + outDegree[v]

        for edge in range(startW,endW):
            my_succ = dest[edge] # my successor
            my_succ_color = colors[my_succ] # my successor's color

            # keep edge if colors are different
            if my_color != my_succ_color:
                supervertex_id = new_vertex[my_color]
                succ_supervertex_id = new_vertex[my_succ_color] # supervertex id of my succ color

                pointer = top_edge[supervertex_id] # where to add edge
                
                new_dest[pointer] = succ_supervertex_id
                new_weight[pointer] = weight[edge]
                new_edge_id[pointer] = edge_id[edge]

                top_edge[supervertex_id] += 1 # increment pointer of current supervertex

@numbapro.jit(numbapro.boolean(numbapro.int32[:]))
def checkConvergenceNumba(outDegree):
    """Return true if all elements of input are 0."""
    n_components = outDegree.size
    for i in range(n_components):
        if outDegree[i] != 0:
            return False
    return True


def exprefixsum(masks, indices, init = 0, nelem = None):
    """
    exclusive prefix sum
    """
    nelem = masks.size if nelem is None else nelem

    carry = init
    for i in xrange(nelem):
        indices[i] = carry
        if masks[i] != 0:
            carry += masks[i]

    #indices[nelem] = carry
    return carry

@numbapro.jit(numbapro.int32(numbapro.int32[:],numbapro.int32[:],numbapro.int32))
def exprefixsumNumba(in_ary, out_ary, init = 0):
    """
    exclusive prefix sum
    """
    nelem = in_ary.size

    carry = init
    for i in range(nelem):
        out_ary[i] = carry
        carry += in_ary[i]

    return carry