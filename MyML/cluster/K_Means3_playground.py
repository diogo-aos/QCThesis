#
# Under work and depricated work.
#
#
#


#
#
#  kernel that solves multiple k-means at the same time, i.e. more than one set of centroids
#

    # data, centroids, labels
    @numbapro.cuda.jit("void(float32[:,:], float32[:,:], int32[:], float32[:], int32[:])")
    def _cu_label_kernel_dists_multiple(data, centroids, labels, dists, indices):

        """
        Computes the labels of each data point storing the distances.
        indices point to the start of the centroids block

        The point of this kernel is to allow for the processing of multiple centroids and
        thus solving multiple problems at the same time. The idea is to receive a 2-dimensional 
        array for the centroids and an indices array. The i-th position of the indices array contains
        the index of a new centroid block, i.e. if indices[1]=5 it means that the first centroid block
        has 5 centroids and the new block starts at centroids[5].

        Not finished. Not implemented.
        """

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

        N = c.shape[0] # number of datapoints
        K, D = b.shape # centroid shape

        if n >= N:
            return

        for i in range(indices.size):
            p = 0

            # first iteration outside loop
            dist = 0.0
            for d in range(D):
                diff = data[n,d] - centroids[0,d]
                dist += diff ** 2

            best_dist = dist
            best_label = -3

            while(p < indices[i+1]):



                # remaining iterations
                for k in range(1,K):

                    dist = 0.0
                    for d in range(D):
                        diff = data[n,d]-centroids[k,d]
                        dist += diff ** 2


                    if dist < best_dist:
                        best_dist = dist
                        best_label = k

                labels[n] = best_label
                dists[n] = best_dist
 

#  kernel that takes advantage of shared memory

    @numbapro.cuda.jit("void(float32[:,:], float32[:,:], int32[:], float32[:])")
    def _cu_label_kernel_dists_sm(a,b,c,dists):

        s_centroids = cuda.shared.array(shape = b.shape, dtype = b.dtype)

        """
        Computes the labels of each data point storing the distances.
        """

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

        N = c.shape[0] # number of datapoints
        K,D = b.shape # centroid shape

        if n >= N:
            return

        # copy centroids to shared memory, thread #0 of each block
        # can be more efficient to distribute copying by multiple threads
        # have to be careful because of warp size
        if tx == 0:
            for k in range(K):
                for d in range(D):
                    s_centroids[k,d] = b[k,d]

        cuda.syncthreads()

        # first iteration outside loop
        dist = 0.0
        for d in range(D):
            diff = a[n,d] - s_centroids[0,d]
            dist += diff ** 2


        best_dist = dist
        best_label = 0

        # remaining iterations
        for k in range(1,K):

            dist = 0.0
            for d in range(D):
                diff = a[n,d] - s_centroids[k,d]
                dist += diff ** 2


            if dist < best_dist:
                best_dist = dist
                best_label = k

        c[n] = best_label
        dists[n] = best_dist