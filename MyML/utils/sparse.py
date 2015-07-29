import numpy as np

from numba import jit
from scipy.sparse import csr_matrix

#from MyML.cluster.linkage import binary_search, binary_search_interval

class EAC_CSR():

    def __init__(self, n_samples=None, max_assocs=None, sort_mode="numpy",
    			 dtype=np.uint8):
        self.n_samples = n_samples
        
        if max_assocs is not None:
            self.max_assocs = max_assocs
        else:
            #self.indptr = np.empty(n_samples, dtype=np.int32)
            raise Exception("Not implemented. Max assocs must be supplied.")

        self.indices = np.empty((n_samples * max_assocs), dtype=np.int32)
        self.data = np.empty((n_samples * max_assocs), dtype=dtype)

        self.indptr = np.arange(n_samples + 1, dtype=np.int32) * self.max_assocs
        self.degree = np.zeros(n_samples + 1, dtype=np.int32)

        self.update_cluster_function = update_cluster
        self.sort_mode = sort_mode

        self.nnz = 0

    def _update_ensemble(self, ensemble):
        # choose the first partition to be the one with least clusters (more 
         # samples per cluster)
        first_partition = np.argmin(map(len,ensemble))
        self.update_first_partition(ensemble[first_partition])
        for p in xrange(len(ensemble)):
            if p == first_partition:
                continue
            self.update_partition(ensemble[p])

    def update_first_partition(self, partition):
        for cluster in partition:
            update_cluster_fp(self.indices, self.data, self.indptr,
                              self.degree, cluster)
            self.nnz += (cluster.size - 1) * cluster.size
            #self.min_assocs = self.degree.min()

    def update_partition(self, partition):
    	update_fn = self.update_cluster_function
    	for cluster in partition:
    		new_nnz = update_fn(self.indices, self.data, self.indptr,
                                self.degree, cluster, self.max_assocs)
    		self.nnz += new_nnz
    	
        if self.sort_mode == "numpy":
        	self._sort_indices()

    def todense(self):
        n = self.n_samples
        return csr_matrix((self.data, self.indices, self.indptr),
                           shape=(n, n)).todense()

    def tocsr(self):
        n = self.n_samples
        return csr_matrix((self.data, self.indices, self.indptr), shape=(n, n))

    def _condense(self):
        nnz = self.nnz
        condense_eac_csr(self.indices, self.data, self.indptr, self.degree)
        self.indices = self.indices[:nnz]
        self.data = self.data[:nnz]
        self.indptr = self.degree
        self.indptr[-1] = nnz
        del self.degree

    def _sort_indices(self):
        # sort all rows by indices
        for row in xrange(self.n_samples):
            start_i = self.indptr[row] # start index
            # start_i = row * self.max_assocs # deduced start index
            end_i = start_i + self.degree[row] # end index
            asorted = self.indices[start_i:end_i].argsort() # get sorted order

            # update data and indices with sorted order
            self.data[start_i:end_i] = self.data[start_i:end_i][asorted]
            self.indices[start_i:end_i] = self.indices[start_i:end_i][asorted]

def _compute_max_assocs_from_ensemble(ensemble):
    return max([max(map(np.size,p)) for p in ensemble])


#
# degree will be the indptr of the condensed CSR
#
@jit(nopython=True)
def condense_eac_csr(indices, data, indptr, degree):
    ptr = degree[0]
    n_samples = degree.size - 1
    degree[0] = 0
    for i in range(1, n_samples):
        i_ptr = indptr[i]
        stopcond = i_ptr + degree[i]
        degree[i] = ptr
        while i_ptr < stopcond:
            indices[ptr] = indices[i_ptr]
            data[ptr] = data[i_ptr]
            ptr += 1
            i_ptr += 1


#
# update cluster of first partition
#
@jit(nopython=True)
def update_cluster_fp(indices, data, indptr, degree, cluster):
    for i in range(cluster.size):
        n = cluster[i] # sample id
        fa = indptr[n] # index of first assoc

        # update number of associations of n with cluster size minus 1
        # to exclude self associations
        degree[n] = cluster.size - 1

        add_ptr = 0
        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            # add association
            data[fa + add_ptr] = 1
            indices[fa + add_ptr] = na
            add_ptr += 1


#
# update cluster of any partition; discards associations that exceed
# pre-allocated space for each sample (=max_assocs)
#
@jit(nopython=True)
def update_cluster(indices, data, indptr, degree, cluster, max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0

    for i in range(cluster.size):
        n = cluster[i] # sample id
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + degree[n] # index of last assoc
        new_n_degree = degree[n]

        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    continue

                # update number of associations of n
                new_n_degree += 1                

                # index to add new association
                new_assoc_ind = fa + new_n_degree - 1

                data[new_assoc_ind] = 1
                indices[new_assoc_ind] = na

        # update number of new non zero elements
        nnz += new_n_degree - degree[n]

        # update with all the new associations
        degree[n] = new_n_degree

    return nnz

@jit(nopython=True)
def update_cluster_sorted(indices, data, indptr, degree, cluster, max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0

    new_assocs_ids = np.empty(max_assocs, dtype=np.int32)
    new_assocs_idx = np.empty(max_assocs, dtype=np.int32)
    # new_assocs_idx_f = np.empty(max_assocs - new_n_degree, dtype=np.int32)

    for i in range(cluster.size):
        n = cluster[i] # sample id
        n_degree = degree[n]
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + n_degree # index of last assoc
        new_n_degree = n_degree

        new_assocs_ptr = 0

        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    continue

                new_assocs_ids[new_assocs_ptr] = na
                new_assocs_idx[new_assocs_ptr] = -(ind + 1)

                # update number of associations of n
                new_assocs_ptr += 1
                new_n_degree += 1

        # nothing else to do if no new association was created
        if new_assocs_ptr == 0:
            continue

        ## make sorted
        # get final idx for new assocs
        # for inc in xrange(new_assocs_ptr):
        #     new_assocs_idx_f[new_assocs_ptr] = new_assocs_idx[new_assocs_ptr] + inc

        # sort
        n_ptr = new_assocs_ptr - 1
        i_ptr = fa + (n_degree - 1)
        o_ptr = fa + n_degree + new_assocs_ptr - 1
        last_index = i_ptr

        while o_ptr >= fa:
            if n_ptr < 0:
                indices[o_ptr] = indices[i_ptr]
                data[o_ptr] = data[i_ptr]
                o_ptr -= 1
                i_ptr -= 1
                continue

            idx = new_assocs_idx[n_ptr]

            # insert new assocs at end
            if idx > last_index:
                indices[o_ptr] = new_assocs_ids[n_ptr]
                data[o_ptr] = 1
                o_ptr -= 1
                n_ptr -= 1
            # add original assocs
            elif i_ptr >= idx:
                
                # try:
                #     indices[o_ptr] = indices[i_ptr]
                # except:
                #     print "i:", i
                #     print "optr:", o_ptr
                #     print "iptr:", i_ptr
                #     raise Exception
                indices[o_ptr] = indices[i_ptr]
                data[o_ptr] = data[i_ptr]
                o_ptr -= 1
                i_ptr -= 1
            # add new assoc
            else:
                indices[o_ptr] = new_assocs_ids[n_ptr]
                data[o_ptr] = 1
                o_ptr -= 1
                n_ptr -= 1

        # update number of new non zero elements
        nnz += new_assocs_ptr

        # update with all the new associations
        degree[n] = new_n_degree

    return nnz


@jit(nopython=True)
def update_cluster_sorted_simple(indices, data, indptr, degree, cluster, max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0

    new_assocs_ids = np.empty(max_assocs, dtype=np.int32)

    for i in range(cluster.size):
        n = cluster[i] # sample id
        n_degree = degree[n]
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + n_degree # index of last assoc
        new_n_degree = n_degree

        new_assocs_ptr = 0

        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    continue

                new_assocs_ids[new_assocs_ptr] = na

                # update number of associations of n
                new_assocs_ptr += 1
                new_n_degree += 1

        # nothing else to do if no new association was created
        if new_assocs_ptr == 0:
            continue

        ## make sorted

        # sort
        n_ptr = new_assocs_ptr - 1
        i_ptr = fa + (n_degree - 1)
        o_ptr = fa + n_degree + new_assocs_ptr - 1
        last_index = i_ptr

        n_ptr_id = new_assocs_ids[n_ptr]
        i_ptr_id = indices[i_ptr]

        while o_ptr >= fa:

        	# second condition for when all new assocs have been added
        	# and only old ones remain
            if i_ptr_id > n_ptr_id or n_ptr < 0:
                indices[o_ptr] = i_ptr_id
                i_ptr -= 1
                i_ptr_id = indices[i_ptr]
            else:
                indices[o_ptr] = n_ptr_id
                n_ptr -= 1
                if n_ptr >= 0:
                    n_ptr_id = new_assocs_ids[n_ptr]
            o_ptr -= 1

        # update number of new non zero elements
        nnz += new_assocs_ptr

        # update with all the new associations
        degree[n] = new_n_degree

    return nnz

#
# surgical because there are no branches while sorting
#
@jit(nopython=True)
def update_cluster_sorted_surgical(indices, data, indptr, degree, cluster,
                                   max_assocs):
    # maximum number of associations pre-allocated for each sample
    nnz = 0

    new_assocs_ids = np.empty(max_assocs, dtype=np.int32)
    new_assocs_idx = np.empty(max_assocs, dtype=np.int32)

    for i in range(cluster.size):
        n = cluster[i] # sample id
        n_degree = degree[n]
        fa = indptr[n] # index of first assoc
        # fa = n * max_assocs # deduce index of first assoc from max_assocs        
        la = fa + n_degree # index of last assoc
        new_n_degree = n_degree

        new_assocs_ptr = 0

        for j in range(cluster.size):
            # exclude self associations
            if i == j:
                continue

            na = cluster[j] # sample id of association
            
            ind = binary_search_interval(na, indices, fa, la)

            # if association exists, increment it
            if ind >= 0:
                curr_assoc = data[ind]
                data[ind] = curr_assoc + 1
            # otherwise add new association
            else:
                # ignore new associations if required to have more associations
                # than those pre-allocated each sample
                if new_n_degree >= max_assocs:
                    continue

                new_assocs_ids[new_assocs_ptr] = na
                new_assocs_idx[new_assocs_ptr] = -(ind + 1)

                # update number of associations of n
                new_assocs_ptr += 1
                new_n_degree += 1

        # nothing else to do if no new association was created
        if new_assocs_ptr == 0:
            continue

        ## make sorted

        # shift original indices
        n_shifts = new_assocs_ptr - 1
        while n_shifts >= 1:
            end_idx = new_assocs_idx[n_shifts]
            start_idx = new_assocs_idx[n_shifts - 1]
            for idx in range(start_idx, end_idx):
                indices[idx + n_shifts] = indices[idx]
            n_shifts -= 1

        # copy new assocs
        new_ptr = 0
        while new_ptr < new_assocs_ptr:
            insert_idx = new_assocs_idx[new_ptr]
            indices[insert_idx] = new_assocs_ids[new_ptr]
            new_ptr += 1

        # update number of new non zero elements
        nnz += new_assocs_ptr

        # update with all the new associations
        degree[n] = new_n_degree

    return nnz




@jit(nopython=True)
def binary_search_interval(key, ary, start, end):
    """
    Inputs:
        key         : value to find
        ary         : sorted arry in which to find the key
        start, end     : interval of the array in which to perform the search
    Outputs:
        if the search was successful the output is a positive number with the
        index of where the key exits in ary; if not the output is a negative
        number; the symmetric of that number plus 1 is the index of where the
        key would be inserted in such a way that the array would still be sorted

    """
    imin = start
    imax = end

    while imin < imax:
        imid = (imax + imin) / 2
        imid_val = ary[imid]

        # key is before
        if key < imid_val:
            imax = imid
        # key is after
        elif key > imid_val:
            imin = imid + 1
        # key is between first edge of imid and next first edge
        else:
            return imid
    return -imin - 1