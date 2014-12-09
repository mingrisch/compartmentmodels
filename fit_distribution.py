import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from compartmentmodels.compartmentmodels import loaddata, savedata
from compartmentmodels.compartmentmodels import CompartmentModel


def synthetic_data(startdict = {'F': 51.0, 'v': 11.2}):
    """this function loads a datafile and calculates a synthetic curve.
    The data provides time and aif.
    
    Parameters:
    -----------
    F, v: float
        initial start values, will be transformed for further calculations
        stored in startdict
        
    returns:
    --------
    time, curve, aif, startdict
    """
    
    time, prebolus, mainbolus = loaddata(filename='compartmentmodels/tests/cerebralartery.csv')
    aif=prebolus-prebolus[0:5].mean()    
    model = CompartmentModel(time=time, curve=aif, aif=aif, startdict=startdict)
    # calculate a model curve
    model.curve = model.calc_modelfunction(model._parameters)
    curve=model.curve
    return time, curve, aif, startdict

def montecarlo_distribution(time, curve, aif, startdict):
    """this functions performs a montecarlo simulation by fitting k different 
    curves which differ by their randomly calculated additional background noise.
    This shows the distribution of the fit parameters.
    
    Parameters:
    -----------
    k: int
        number of fitting runs
    noise_intensity: float
        intensity of additional background noise
        
    returns:
    --------
    fit_dist: ndarray
        distribution of the fit parameters
    """
    
    k=1000
    noise_intensity = 0.002 * aif.max()
    model= CompartmentModel(time=time, curve=curve, aif=aif, startdict=startdict)
    fit_dist = np.zeros((3,k))  
    for i in range(k):
        model.curve = noise_intensity * np.random.randn(len(time)) + curve
        model.fit_model()
        
        parameters=model.get_parameters()
        fit_dist[:,i] = (parameters['F'], parameters['v'], parameters['MTT'])
    return fit_dist
    
def bootstrap_distribution(time, curve, aif,startdict):
    """performs the bootstrap.
    returns bootstrap distribution.
    """
    model= CompartmentModel(time=time, curve=curve, aif=aif, startdict=startdict)
    model.curve += 0.002 * aif.max() * np.random.randn(len(time))

    model.fit_model()
    model.bootstrap(k=1000)
    bootstrap_dist = model.bootstrap_result_physiological
    return bootstrap_dist
    
if __name__=="__main__":
    time, curve, aif, startdict = synthetic_data()
    
    # monte carlo 
    mc_results = montecarlo_distribution(time, curve, aif, startdict)

    # bootstrapping
    b_results = bootstrap_distribution(time,curve,aif,startdict)

    np.savetxt('mc_results', mc_results)
    np.savetxt('b_results', b_results)

