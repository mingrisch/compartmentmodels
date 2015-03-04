import numpy as np
def c_convolution_exp(time, aif, lamda):
    """ Calculate the discrete convolution of aif with an exponential with time constant -lamda"""
    t=time
    x=aif
    y=np.zeros_like(t)
    for i in range(1, len(y)):
        dt = t[i] - t[i - 1]
        edt = np.exp(-lamda * dt)
        m = (x[i] - x[i - 1]) / dt
        y[i] = edt * y[i - 1] + (x[i - 1] - m * t[i - 1]) / lamda * (
            1 - edt) + m / lamda / lamda * ((lamda * t[i] - 1) - edt * (lamda * t[i - 1] - 1))
    return y
