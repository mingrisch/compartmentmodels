# cython: profile=True
import numpy as np
from scipy import optimize

from compartmentmodels.GenericModel import GenericModel
        
if __name__ == '__main__':
    # create  a boxcar function for the aif and calculate some convolutions
    # with an exponential, as test case for further implementations of this
    # discrete convolution
    
    npt=200
    tmax=50

    time = np.linspace(0,tmax,num=npt)
    
    aif = np.zeros_like(time)
    aif[(time<10) & (time >1)] = 1

    curve = np.zeros_like(time)
    
    gp = GenericModel(time, curve, aif)

    # generate some convolutions of this 
    lambdalist =[3.2, 1.8, 0.2, 0.0000110]

    # set up an array for the outfile
    fileheader=['aif']
    for el in lambdalist:
        fileheader.append(str(el))
                
    outarray=np.zeros((len(fileheader),npt))

    fileheader=','.join(fileheader)
    
    for i, lam in enumerate(lambdalist):
        curve=gp.convolution_w_exp(lam)
        outarray[i+1,:]=curve
        print "Mean of convolution with {}: {}".format(lam, curve.mean())

    saved=np.savetxt('convolutions.csv', outarray.transpose(), header=fileheader)

