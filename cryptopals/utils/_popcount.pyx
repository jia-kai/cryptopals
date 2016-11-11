# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

cdef extern from "_popcount.h":
    unsigned _do_popcount(uint8_t *, size_t) nogil

def popcount8(arr):
    """popcount for uint8 datatype"""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] arrt = arr
    return _do_popcount(<uint8_t*>arrt.data, arrt.size)
