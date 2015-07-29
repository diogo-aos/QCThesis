import numpy as np
from numba import jit

#
# jitted version was 110 times faster than unjitted for 1e6 array
# ported and adapted to arg-k-select from:
# http://blog.teamleadnet.com/2012/07/quick-select-algorithm-find-kth-element.html
@jit(nopython=True)
def arg_k_select(ary, k, out):
# def arg_k_select(ary, k):
    args = np.empty(ary.size, dtype=np.int32)
    for i in range(args.size):
        args[i] = i

    fro = 0
    to = ary.size - 1

    while fro < to:
        r = fro
        w = to
        mid_arg = args[(r+w) / 2]
        mid = ary[mid_arg]

        while r < w:
            r_arg = args[r]
            w_arg = args[w]
            if ary[r_arg] >= mid:
                tmp = args[w]
                args[w] = args[r]
                args[r] = tmp
                w -= 1
            else:
                r += 1

        r_arg = args[r]
        if ary[r_arg] > mid:
            r -= 1

        if k <= r:
            to = r
        else:
            fro = r + 1

    for i in range(k):
        out[i] = args[i]

    # return args[:k]


def quicksort(array):
    _quicksort(array, 0, len(array) - 1)
 
def _quicksort(array, start, stop):
    if stop - start > 0:
        pivot, left, right = array[start], start, stop
        while left <= right:
            while array[left] < pivot:
                left += 1
            while array[right] > pivot:
                right -= 1
            if left <= right:
                array[left], array[right] = array[right], array[left]
                left += 1
                right -= 1
        _quicksort(array, start, right)
        _quicksort(array, left, stop)


