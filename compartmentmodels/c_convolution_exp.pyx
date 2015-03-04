import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport exp

DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
def c_convolution_exp(np.ndarray[DTYPE_t,ndim=1] t, np.ndarray[DTYPE_t,ndim=1] x, double lamda):
    """ Calculate the discrete convolution of aif with an exponential with time constant -lamda"""
    #y=np.zeros_like(t)
    cdef int ny = t.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.zeros(ny, dtype=np.float64)

    cdef int i
    cdef double edt, dt, m
    
    for i in range(1, ny ):
        dt = t[i] - t[i - 1]
        edt = exp(-lamda * dt)
        m = (x[i] - x[i - 1]) / dt
        y[i] = edt * y[i - 1] + (x[i - 1] - m * t[i - 1]) / lamda * (
            1 - edt) + m / lamda / lamda * ((lamda * t[i] - 1) - edt * (lamda * t[i - 1] - 1))
    return y
