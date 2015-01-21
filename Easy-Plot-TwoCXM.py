from compartmentmodels.compartmentmodels import loaddata, savedata
from compartmentmodels.compartmentmodels import TwoCXModel
import numpy as np
import matplotlib.pyplot as plt
import itertools
# need to set a stardict and convert input parameter maybe. 
#Set parameters for the model

#y=(np.array(range(1,100)))/10.
#s=[y,y,y,y]
#a,b,c,d = list(itertools.product(*s))
a, b, c, d = ( 25.0, 4.2, 0.000001, 13.1)

startdict = {'FP': a, 'VP': b, 'PS':c,'VE':d}
time, aif1, aif2 =loaddata(filename='tests/cerebralartery.csv')  

aif = aif1 - aif1[0:5].mean()

model = TwoCXModel(time=time,curve=aif, aif=aif,startdict=startdict)

# calculate a model curve
model.curve = model.calc_modelfunction(model._parameters)
model.curve += 0.02 * model.curve.max() * np.random.randn(len(time))
# number of bootstraps
#model.k=100  

fitresult = model.fit_model()

plt.plot(model.time, model.curve)
plt.plot(model.time, model.fit)
plt.show()