# cython: profile=True
import numpy as np
import scipy as sp
from scipy import optimize, signal


class GenericModel:

    """This is a base class for all models. It provides the general
    framework and contains a fit function for a simple one compartment
    model. For other models, we recommend subclassing this class and
    overwriting/redefining the function calc_residuals(self,
    parameters)



    Attributes
    ----------

    time: np.ndarray
        the time axis

    aif: np.ndarray:
        the arterial input function

    curve: np.ndarray
        the (measured) curve to which the model should be fitted

    residuals: np.ndarray
        residual vector after a fit

    fit: np.ndarray
        the curve as described by the model (after a fit)

    aic: float
        Akaike information criterion, after a fit

    rc: int
        return code of the fit routine

    parameters: np.ndarray
        array of raw parameters
        todo: should be changed to _parameters, only internal us

    _cython_available: bool
        do we have cython, or use it?


    _fitted: bool
        has a fit been performed?

    readable_parameters: dict
        Dictionary of readable parameters




    """

    def __init__(self, time=sp.empty(1), curve=sp.empty(1), aif=sp.empty(1)):
        # todo: typecheck for time, curve, aif
        # they should be numpy.ndarrays with the same size and dtype float
        self.time = time
        self.aif = aif
        self.curve = curve
        self.residuals = curve
        self.fit = sp.zeros_like(curve)
        self.aic = 0.0
        self.rc = 0
        self._parameters = np.zeros(2)

        self._cythonavailable = False
        self._fitted=False

        # needed for calculation of the Akaike information criterion:
        self.nooffreeparameters = 2

        # this dictionary will contain human-readable (parameter,value) entries
        self.readable_parameters = {'F': 0.0,
                                    'v': 0.0,
                                    'MTT': 0.0}

    def __str__(self):
        return "Generic model"

    # get functions
    def get_parameters(self):
        """Return a dictionary of fitted model parameters.

        To be used after a successful fit.
        Converts the 'raw' fit parameters to the 'physiological' model
        parameters and saves them in self.readable_parameters.
        Parameters
        ----------
        None

        Returns
        -------

        dict
            Dictionary containing the estimated model parameters

        Notes
        -----

        For the one-compartment model, these parameters aris a flow, a volume
        and the corresponding transit time. Derived models will likely have to
        override this method.  """

        FP = self._parameters[0] * 6000.
        VP = self._parameters[0] / self._parameters[1] * 100
        TP = 1 / self._parameters[1]
        self.readable_parameters["F"] = FP
        self.readable_parameters["v"] = VP
        self.readable_parameters["MTT"] = TP

        return self.readable_parameters

    def get_raw_parameters(self):
        return self._parameters

    def get_aif(self):
        return self.aif

    def get_curve(self):
        return self.curve

    def get_time(self):
        return self.time

    def get_residuals(self):
        return self.residuals

    def get_fit(self):
        return self.fit

    # set functions:
    def set_parameters(self, newparameters):
        self._parameters = newparameters

    def set_time(self, newtime):
        self.time = newtime

    def set_curve(self, newcurve):
        self.curve = newcurve

    def set_aif(self, newaif):
        self.aif = newaif

    # convolution of aif with an exponential
    def convolution_w_exp(self, lamda, fftconvolution=False):
        """ returns the convolution of self.aif with an exponential
         exp(-lamda*t). we follow the notation introduced in
        http://edoc.ub.uni-muenchen.de/14951/
        """
        # todo: check for lamda == zero: in this case, convolve with
        # constant, i.e. intvector.
#        y=cy_conv_exp(list(self.time),list(self.curve),list(self.aif),tau)
        if fftconvolution:
            # calculate the convolution via fft
            expon = np.exp(-lamda * self.time)
            # y = signal.fftconvolve(self.aif, expon, mode='same')
            y = np.convolve(self.aif, expon, mode='full')
            # we need to scale down by a factor dt
            y = y * (self.time[1] - self.time[0])
            # and we need the first half only:
            y = y[0:len(y) / 2 + 1]

            return y

        elif self._cythonavailable:
            # calculcate the discrete  convolution with the cpython
            # implementation
            pass
            # y = cy_conv_exp(self.time, self.curve, self.aif, lamda)
        else:
            # calculate the discrete convolution in pure python

            # for i in range(1,N):
            #     dt=t[i]-t[i-1]
            #     edt=np.exp(-lam*dt)
            #     m=(x[i]-x[i-1])/dt
            #     y[i]=edt*y[i-1]+(x[i-1]-m*t[i-1])/lam*(1-edt)+m/lam/lam*((lam*t[i]-1)-edt*(lam*t[i-1]-1))
            # return y
            # this is a 1:1 port from the cython version.
            # maybe this could be written more efficiently as a generator
            t = self.time
            x = self.aif
            y = np.zeros_like(t)

            for i in range(1, len(y)):
                dt = t[i] - t[i - 1]
                edt = np.exp(-lamda * dt)
                m = (x[i] - x[i - 1]) / dt
                y[i] = edt * y[i - 1] + (x[i - 1] - m * t[i - 1]) / lamda * (
                    1 - edt) + m / lamda / lamda * ((lamda * t[i] - 1) - edt * (lamda * t[i - 1] - 1))
            return y
        # return sp.asarray(y)

    def intvector(self):
        """This function calculates the convolution of the arterial
        input function with a constant. Basically, this is an
        integration of the AIF."""
        y = self.aif
        x = self.time
        # no need for a for-loop

        ysum = y[1:] + y[0:-1]
        # dx=x[1:]-x[0:-1]
        dx = sp.diff(x)
        integral = sp.zeros_like(x)
        integral[1:] = sp.cumsum(ysum * dx / 2)

        return integral

    def calc_modelfunction(self, parameters):
        """ Calculate the model curve for given parameters

        Paraemters
        ----------
        parameters: numpy.ndarray
            model parameters

        Returns:
        --------
        np.ndarray
            an array with the model values

        """
        modelcurve=parameters[0] * self.convolution_w_exp(parameters[1])

        return modelcurve

    def _calc_residuals(self, parameters):
        """ Wrapper around calc_modelfunction

        This function wraps around the model function so that it can be called
        from scipy.optimize.minimize, i.e. it accepts an array of fit parameters and
        returns a scalar, the sum of squared residuals
        """

        residuals=self.curve-self.calc_modelfunction(parameters)
        self.residuals=residuals
        return np.sum(residuals**2)

    def calc_residuals(self, parameters, fjac=None):
        """Deprecated. (was used for the mpfit fitting).
        
        This function calculates the residuals for a
        one-compartment model with residual function
        p[0]*exp(-t*p[1]).  self.residuals is set to the resulting
        array, furthermore, the sum of resulting array is returned.  Note:
        This function will be called from the solver quite often. For
        ptimizing performance, someone could rewrite it in c or
        fortran. For now, we're happy with this implementation """

        residuals = self.curve - \
            (parameters[0] * self.convolution_w_exp(parameters[1]))
        self.residuals = residuals
        status = 0
        # return squared sum of res.
        return ([status, residuals])

    def convert_startdict(self, startdict):
        """
        Take a dictionary containing start values for FP,VP,PS,VE and
        calculate start values for the fitting. Save them in an array
        as required by calc_residuals. This function is meant to be
        implemented by each model.
        """
        if not type(startdict).__name__ == 'dict':
            return
        # easy for the generic model:
        FP = startdict.get("FP") / 6000.  # we need to convert to SI
        VP = startdict.get("VP") / 100
        lamda = 1. / (VP / FP)
        return [FP, lamda]

    def fit_model(self, startdict,
                  constrained=True):
        """ Perform the model fitting. 
        
        this function attempts to fit the model to the curve, using
        the startparameters as initial value.
        We use scipy.optimize.minimize and use sensible bounds for the
        parameters.
        """
        startparameters = self.convert_startdict(startdict)
        if constrained:
            bounds=[(0,None), (0,None)]
            method='L-BFGS-B'
        else:
            bounds=[(None,None), (None,None)]
            method='BFGS'

        fit_results=minimize(self._calc_residuals, startparameters,
                method=method,
                bounds=bounds)

        self._parameters=fit_results.x
        self.fit=self.calc_modelfunction(self._parameters)

        return fit_results.success

    def get_AIC(self):
        """this method returns the corrected Akaike information
        criterion (Glatting 07).  It is only available after a
        successful fit"""
        if self.rc > 0:
            n = self.fit.size
            aic = n * sp.log(sp.sum(sp.square(self.residuals)) / n)
            +2 * (self.nooffreeparameters + 1)
            +2 * (self.nooffreeparameters + 1) * \
                (self.nooffreeparameters + 2) / \
                (n - self.nooffreeparameters - 2)
            return aic
        else:
            return False


if __name__ == '__main__':
    import pylab
    npt = 200
    tmax = 50

    # inputs. one could, in principle, load this dataset.
    # but we don't have one yet.

    time = sp.linspace(0, tmax, num=npt)
    aif = sp.square(time) * sp.exp(-time / 2)
    curve = aif * 0.0
    # aif=aif.astype('float64')
    # curve=curve.astype('float64')
    gptmp = GenericModel(time, curve, aif)

    # generate an aif with recirculation:
    recirculation1 = gptmp.convolution_w_exp(1. / 8.) / 8
    recirculation1 = sp.roll(recirculation1, 10 * 4)
    recirculation1[0:10 * 4] = 0
    recirculation2 = gptmp.convolution_w_exp(100.) / 100
    recirculation2 = sp.roll(recirculation2, 20 * 4)
    recirculation2[0:20 * 4] = 0
    aif = aif + recirculation1 + recirculation2
    gptmp.set_aif(aif)

    startdict = {"FP": 45.0, "VP": 10.0}
    true_parameters = gptmp.convert_startdict(startdict)
    curve = true_parameters[0] * gptmp.convolution_w_exp(true_parameters[1])
    max_curve = sp.amax(curve)
    noise = 0.05 * max_curve * (sp.rand(npt) - 0.5)
    curve = curve + noise
# ____ end of input calculations

    # set up the generic model object.
    gp = GenericModel()
    gp.set_time(time)
    gp.set_curve(curve)
    gp.set_aif(aif)
    # initial parameters for the fit
    start_parameters = sp.array([80. / 6000, 7.])
    startdict = {"FP": 50.0, "VP": 12.0}

    # fit the model to the curve
    gp.fit_model(startdict)
    # text output

    print 'True parameters: ', true_parameters
    print 'Initial parameters: ', start_parameters
    print 'Estimated parameters: '

    pardict = gp.get_parameters()
    for k, v in pardict.items():
        print k, v
    print 'AIC: ',  gp.get_AIC()

    # graphical output
    pylab.plot(gp.get_time(), gp.get_curve(), 'bo')
    pylab.plot(gp.get_time(), gp.get_fit(), 'g-')
    pylab.title("One compartment fit")
    pylab.xlabel("time [s]")
    pylab.ylabel("concentration [a.u.]")
    pylab.legend(('simulated curve', 'model fit'))
    pylab.show()
