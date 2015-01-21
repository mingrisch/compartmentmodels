from compartmentmodels.compartmentmodels import *
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set_context('notebook')
sns.set_style("whitegrid")
sns.set_palette('colorblind')

startdict = {'F': 51.0, 'v': 11.2}
time, prebolus, mainbolus = loaddata(filename='tests/cerebralartery.csv')
aif=prebolus-prebolus[0:5].mean()    
model = CompartmentModel(time=time, curve=aif, aif=aif, startdict=startdict)
# calculate a model curve
model.curve = model.calc_modelfunction(model._parameters)
model.curve += 0.002 * aif.max() * np.random.randn(len(time))
model.fit_model()
fit=model.fit 
curve=model.curve



residuals_bootstrap = curve - fit

# shapiro-wilk test for normaly distributed residuals
w,p = sp.stats.shapiro(residuals_bootstrap)
print 'test statistic:',w,'p-value:', p
if p < 0.05:
    raise ValueError('probably not normal distributed residuals. Try another model')
                

# adds +0.1 to the left and -0.1 to the right side of the residuals. makes it not normal distr.
# just for evaluation...
residuals_bootstrap1 = residuals_bootstrap[0:len(residuals_bootstrap)/2]+1.5
residuals_bootstrap2= residuals_bootstrap[len(residuals_bootstrap)/2:len(residuals_bootstrap)]-1.5
residuals_bootstrap12=np.append(residuals_bootstrap1,residuals_bootstrap2)



f, (ax1,ax2, ax3) = plt.subplots(1,3, figsize=(12.5,5))

ax1.plot(time, residuals_bootstrap, label='original residuals')
ax1.plot(time, residuals_bootstrap12, label='modified residuals')
ax1.plot(time, curve, alpha=0.3, label='curve')
ax1.plot(time,fit, alpha=0.3, label='fit')
ax1.set_title('Curve, fit, residuals')
ax1.set_xlabel('time')
ax1.set_ylabel('value')
ax1.legend()

sns.distplot(residuals_bootstrap, ax=ax2, label='original residuals')
sns.distplot(residuals_bootstrap12, ax=ax2, label='modified residuals')
ax2.set_title('Distribution of residuals')
ax2.legend()

autocorrelation_plot(residuals_bootstrap,ax=ax3, label='original residuals')
autocorrelation_plot(residuals_bootstrap12,ax=ax3, label='modified residuals')
ax3.set_title('Autocorrelation plot of residuals')
ax3.set_xlabel('Lag')
ax3.set_ylabel('Autocorrelation')
#plt.show()
f.tight_layout()


f.savefig('Autocorrelation.pdf')
        