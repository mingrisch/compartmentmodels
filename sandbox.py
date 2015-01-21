"""
playground script.

"""
import scipy as sp
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

""" prepare a model instance with startdict, time, aif and synthetic curve +
additional background noise, ready for fitting
"""

from compartmentmodels.compartmentmodels import loaddata, savedata
from compartmentmodels.compartmentmodels import CompartmentModel
startdict = {'F': 31.0, 'v': 11.2}

time, aif1, aif2 =loaddata(filename='tests/cerebralartery.csv')    
# remove baseline signal
aif = aif1 - aif1[0:5].mean()
model = CompartmentModel(
    time=time, curve=aif, aif=aif, startdict=startdict)
# calculate a model curve
model.curve = model.calc_modelfunction(model._parameters)
model.curve += 0.05* model.curve.max() * np.random.randn(len(time))

f, (a1, a2) =plt.subplots(1,2)

a1.plot(time, aif)
a2.plot(time, model.curve, label='original curve')


if model.fit_model():
    a2.plot(time, model.fit, label='fit')
    a2.legend()

    model.bootstrap(k=1000)
    print "Successfull fit."
    print "Result dictionary is: "
    print model.get_parameters()
f.savefig('tmpfig.png')
f.show()
