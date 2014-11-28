# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy.optimize import minimize

# helper functions for saving and loading 'model datasets'

def loaddata(filename, separator=',',comment='#'):
    """ This function loads time, curve and AIF arrays from 
    a textfile with the structure:
    \# commentline (lines are ignored if they start with 'comment', 
    default is \#)
    time[0], curve[0], aif[0]
    time[1], curve[1], aif[1]
    ...
    
    loaddata performs sanity checks and raises a Value Error if:
    if time, curve and aif have not the same length
    if time is not monotonously raising


    otherwise, the function returns a tuple (time, curve, aif)
    """
    try:
        t,c,a = np.loadtxt(filename, comments=comment, 
                        delimiter=separator,unpack=True)
    except:
        raise IOError

    #sanity checks:
    if (
        (len(t) != len(a)) or 
        (len(t)!=len(c)) or 
        (len(a) != len(c))):
        raise ValueError

    return (t,c,a)



def savedata(filename,time, curve, aif):
    """This function saves three 1D arrays time, curve and AIF
    into a file that complies to the format accepted by loaddata"""
    try:
        np.savetxt(filename, np.transpose((time,curve,aif)), delimiter=',')
        return True
    except:
        return False

class CompartmentModel:

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

    _parameters: np.ndarray
        array of raw parameters, used internally for the calculation of model
        fits

    _cython_available: bool
        do we have cython, or use it?
        Currently, cython is not used, but this may change. 


    _fitted: bool
        has a fit been performed?

    readable_parameters: dict
        Dictionary of physiological parameters. These can be used as start
        parameters for the fit, and are calculated from self._parameters after
        a successfull fit




    """

    def __init__(self, time=sp.empty(1), curve=sp.empty(1), aif=sp.empty(1),
                 startdict={'F': 50.0, 'v': 12.2}):
        # todo: typecheck for time, curve, aif
        # they should be numpy.ndarrays with the same size and dtype float
        self.time = time
        self.aif = aif
        self.curve = curve
        self.residuals = curve
        self.fit = sp.zeros_like(curve)
        self.aic = 0.0
        self._parameters = np.zeros(2)

        self.k = 500
        self._cythonavailable = False
        self._fitted = False
        self._bootstrapped = False

        # Perform convolution using fft or linear interpolation?
        self.fft = False
        # this dictionary will contain human-readable (parameter,value) entries
        self.readable_parameters = startdict

        # this will store the OptimizeResult object
        self.OptimizeResult = None

        # convert the dictionary entry to 'raw' parameters:
        self._parameters = self.convert_startdict(startdict)
        
        #defy method and bounds, which fit_model will use:
        self.constrained=False
        self.set_constraints(self.constrained)
   
        
    def __str__(self):
        return "Base class for compartment models."

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

        For the one-compartment model, these parameters are a flow, a volume
        and the corresponding transit time. Derived models will likely have to
        override this method.  """

        F = self._parameters[0] * 6000.
        v = self._parameters[0] / self._parameters[1] * 100
        mtt = 1 / self._parameters[1]

        
        self.readable_parameters["F"] = F
        self.readable_parameters["v"] = v
        self.readable_parameters["MTT"] = mtt

        if self._fitted:
            self.readable_parameters["Iterations"] = self.OptimizeResult.nit

        
        if self._bootstrapped:
            # convert bootstrapped_raw to boostrapped_physiological
            self.bootstrap_result_physiological=np.zeros((3,self.k))
            for i in range(self.k):
                F_physiological = self.bootstrap_result_raw[0,i] * 6000
                v_physiological = self.bootstrap_result_raw[0,i] / self.bootstrap_result_raw[1,i] * 100
                mtt_physiological = 1 / self.bootstrap_result_raw[1,i]
                self.bootstrap_result_physiological[:,i] = (F_physiological, v_physiological, mtt_physiological)
            

            self.bootstrap_percentile = np.percentile(self.bootstrap_result_physiological, [17, 50, 83], axis=1)
            #self.mean = self.bootstrap_result_physiological.mean(axis=1)
            #self.std = self.bootstrap_result_physiological.std(axis=1)
            #self.low = self.mean - self.std
            #self.high = self.mean + self.std
            
            self.readable_parameters["low estimate"] = {'F':self.bootstrap_percentile[0,0], 'v':self.bootstrap_percentile[0,1], 'MTT':self.bootstrap_percentile[0,2]}
            self.readable_parameters["mean estimate"] = {'F':self.bootstrap_percentile[1,0], 'v':self.bootstrap_percentile[1,1], 'MTT':self.bootstrap_percentile[1,2]}
            self.readable_parameters["high estimate"] = {'F':self.bootstrap_percentile[2,0], 'v':self.bootstrap_percentile[2,1], 'MTT':self.bootstrap_percentile[2,2]}
            
            #self.readable_parameters["low estimate"] = {'F':self.low[0], 'v':self.low[1], 'MTT':self.low[2]}
            #self.readable_parameters["mean estimate"] = {'F':self.mean[0], 'v':self.mean[1], 'MTT':self.mean[2]} 
            #self.readable_parameters["high estimate"] = {'F':self.high[0], 'v':self.high[1], 'MTT':self.high[2]}                  

        #print self.readable_parameters       
        return self.readable_parameters



    def get_raw_parameters(self):
        # I don't think we ever need this function.
        print "Deprecation warning: this function is deprecated and will be removed"
        return self._parameters

    def get_aif(self):
        print "Deprecation warning: this function is deprecated and will be removed"
        return self.aif

    def get_curve(self):
        print "Deprecation warning: this function is deprecated and will be removed"
        return self.curve

    def get_time(self):
        print "Deprecation warning: this function is deprecated and will be removed"
        return self.time

    def get_residuals(self):
        print "Deprecation warning: this function is deprecated and will be removed"
        return self.residuals

    def get_fit(self):
        print "Deprecation warning: this function is deprecated and will be removed"
        return self.fit

    # set functions:
    def set_parameters(self, newparameters):
        """
        I am not sure whether this function should even exist.

        self._parameters should only be calculated internally.
        """
        print "Deprecation warning: this function is deprecated and will be removed"
        self._parameters = newparameters

    def set_time(self, newtime):
        print "Deprecation warning: this function is deprecated and will be removed"
        self.time = newtime

    def set_curve(self, newcurve):
        print "Deprecation warning: this function is deprecated and will be removed"
        self.curve = newcurve

    def set_aif(self, newaif):
        print "Deprecation warning: this function is deprecated and will be removed"
        self.aif = newaif

    # convolution of aif with an exponential
    def convolution_w_exp(self, lamda, fftconvolution=False):
        """ Convolution of self.aif with an exponential.

        This function returns the convolution of self.aif with an exponential
        exp(-lamda*t). Currently, two implementations are available: per
        default, we calculate the convolution analytically with a linearly
        interpolated AIF, following the notation introduced in
        http://edoc.ub.uni-muenchen.de/14951/. Alternatively, convolution via
        fft can be used. Currently, the tests for the FFT convolution that
        compare FFT to standard convolution fail. Presumably, the reason is
        the difference between sinc-interpolation (by FFT) and linear
        interpolation.

        Parameters
        ----------
        lamda: float
            constant in the exponential exp(-lamda*self.time)

        fftconvolution: bool, optional
            Should the convolution be calculated with FFT? Default is false.

        Returns
        -------
        np.array
            an array with the result of the convolution.
        """
        # todo: check for lamda == zero: in this case, convolve with
        # constant, i.e. intvector.
#        y=cy_conv_exp(list(self.time),list(self.curve),list(self.aif),tau)
        if lamda == 0:
            # in this case, we do convolution with a constant, i.e. we call
         # self.intvector
            return self.intvector()
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

        Parameters
        ----------
        parameters: numpy.ndarray
            model parameters

        Returns:
        --------
        np.ndarray
            an array with the model values

        """
        modelcurve = parameters[
            0] * self.convolution_w_exp(parameters[1], fftconvolution=self.fft)

        return modelcurve

    def _calc_residuals(self, parameters, curve):
        """ Wrapper around calc_modelfunction

        This function wraps around the model function so that it can be called
        from scipy.optimize.minimize, i.e. it accepts an array of fit parameters and
        returns a scalar, the sum of squared residuals

        Parameters
        ---------
        parameters:
            raw parameters, e.g. self_parameters

        Returns
        -------
        double
            sum of squared residuals
        """
        
        residuals = curve - self.calc_modelfunction(parameters)
        self.residuals = residuals
        return np.sum(residuals ** 2)


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
            (parameters[
             0] * self.convolution_w_exp(parameters[1],  fftconvolution=self.fft))
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

        Parameters:
        ----------
        startdict: Dictionary
            Dictionary containing the initial values for the model parameters 

        Returns:
        -------
        np.ndarray: 
            array containing the raw model parameters 
        """
        if not type(startdict).__name__ == 'dict':
            return
        # easy for the generic model:
        FP = startdict.get("F") / 6000.  # we need to convert to SI
        VP = startdict.get("v") / 100
        lamda = 1. / (VP / FP)
        
        # store the parameters in self._parameters:
        self._parameters=np.asarray([FP, lamda])
        return np.asarray([FP, lamda])

    def set_constraints(self, constrained):
        """ This function sets the contraints for fitting. 
        """
    
        self.constrained=constrained

        if constrained:
            self.bounds = [(0, None), (0, None)]
            self.method = 'L-BFGS-B'
        else:
            self.bounds = [(None, None), (None, None)]
            self.method = 'BFGS'
            
            
    def fit_model(self, startdict=None,
                  constrained=True, fft=False):
        """ Perform the model fitting. 

        this function attempts to fit the model to the curve, using
        the startparameters as initial value.
        We use scipy.optimize.minimize and use sensible bounds for the
        parameters.

        Parameters:
        ----------
        startdict: dictionary
            Dictionary with initial values
        constrained: bool
            Perform fitting with or without positivity constraints (default True)
        fft: bool
            use fft for the calculation of the convolution (default False)

        Returns:
        -------
        bool
            Fit successful?
        """
        self._fitted = False
        self.fft = fft
        self.OptimizeResult = None
        # convert start dict to self._parameters
        if startdict is None:
            startparameters = self._parameters
        else:
            startparameters= self.convert_startdict(startdict)
        

        self.set_constraints(constrained)
        
        fit_results = minimize(self._calc_residuals, startparameters, args=(self.curve,),
                               method=self.method,
                               bounds=self.bounds)

        self._parameters = fit_results.x

        # store the Optimize Result, in case we need it later on
        self.OptimizeResult = fit_results
        self.fit = self.calc_modelfunction(self._parameters)
        self._fitted = fit_results.success

        #print "Fit returned {} and yielded the parameters {}".format(fit_results.success, fit_results.x)

        return fit_results.success

    def get_AIC(self):
        """this method returns the corrected Akaike information
        criterion (Glatting 07).  It is only available after a
        successful fit"""
        if self._fitted:
            n = self.fit.size
            npar = len(self._parameters)

            aic = n * sp.log(sp.sum(sp.square(self.residuals)) / n)
            +2 * (npar + 1)
            +2 * (npar + 1) * \
                (npar + 2) / \
                (n - npar - 2)
            return aic
        else:
            return False
    
    def bootstrap(self, k=None):
        """ Bootstrap the parameter estimates after a successful fit.

        Parameters:
        ----------
        k : int
            Number of bootstrap runs. Defaults to 500

        Returns:
        -------
        This needs to be discussed. Set the parameter dictionary appropriately?
        Store the bootstrap estimates internally in the class? 
        """

        if not self._fitted:
            return None
        
        if k:
            self.k = k		
        # need to change variables for bootstrapping
        original_curve = self.curve
        original_fit = self.fit
        original_parameters = self._parameters
        original_readable_parameters= self.readable_parameters
        
        # set of residuals calculated for bootstrapping
        residuals_bootstrap = (self.curve - self.fit)
        
        # array, which will be overwritten with the results
        self.bootstrap_result_raw = np.zeros((2,self.k))
        
        # bootstrapping loop
        for i in range(self.k):
            sample_index = np.random.randint(0,residuals_bootstrap.shape[0], residuals_bootstrap.shape)
            self.curve = self.fit + residuals_bootstrap[sample_index]
            
            self.fit_model(self.readable_parameters)
            
            self.bootstrap_result_raw[:,i] = self._parameters
            #self.bootstrap_result[:,i] = self.get_parameters()['v','F']
    
        #self.mean = self.bootstrap_result.mean(axis=1)
        #self.std = self.bootstrap_result.std(axis=1)
        
        # rechange variables
        self.curve= original_curve
        self.fit = original_fit
        self._parameters = original_parameters
        self.readable_parameters = original_readable_parameters
        
        self._bootstrapped=True

        return self.get_parameters()
        #print self.get_parameters()
        # test: does boostrap return a dictionary with appropriate keys and 3-tuples as value?



class CompartmentUptakeModel(CompartmentModel):

    """ A compartment uptake model, as in the Sourbron/Buckley paper.

    This class is derived from the Compartment Model

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

    _parameters: np.ndarray
        array of raw parameters, used internally for the calculation of model
        fits

    _cython_available: bool
        do we have cython, or use it?
        Currently, cython is not used, but this may change. 


    _fitted: bool
        has a fit been performed?

    readable_parameters: dict
        Dictionary of physiological parameters. These can be used as start
        parameters for the fit, and are calculated from self._parameters after
        a successfull fit




    """

    def __init__(self, time=sp.empty(1), curve=sp.empty(1), aif=sp.empty(1),
                 startdict={'Fp': 50.0, 'v': 12.2, 'PS': 2.1}):
        # call __super__.__init__, with the appropriate parameters.

        # A lot of this can simply be copied from previous implementations.

        # Nevertheless, we do this strictly test-driven.

        # other initializations:

        # functions to override:
        pass

    def calc_modelfunction(self, parameters):
        pass

    def _calc_residuals(self, parameters):
        # in fact, this does not need to be redefined
        pass

    def convert_startdict(self, startdict):
        # this needs to be redefined
        pass

    def get_parameters(self):
        # this needs to be reimplemented.

        # Also convert_startdict and get_parameters are, in fact, getters and
        # setters. Maybe the should be renamed in a more consistent fashion.
        pass



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
    gptmp = CompartmentModel(time, curve, aif)

    # generate an aif with recirculation:
    recirculation1 = gptmp.convolution_w_exp(1. / 8.) / 8
    recirculation1 = sp.roll(recirculation1, 10 * 4)
    recirculation1[0:10 * 4] = 0
    recirculation2 = gptmp.convolution_w_exp(100.) / 100
    recirculation2 = sp.roll(recirculation2, 20 * 4)
    recirculation2[0:20 * 4] = 0
    aif = aif + recirculation1 + recirculation2
    gptmp.aif = aif

    true_values = {"F": 45.0, "v": 10.0}
    true_parameters = gptmp.convert_startdict(true_values)
    curve = true_parameters[0] * gptmp.convolution_w_exp(true_parameters[1])
    max_curve = sp.amax(curve)
    # to do: what the hell is this:
    noise = 0.08 * max_curve * (sp.rand(npt) - 0.5)
    curve = curve + noise
# ____ end of input calculations

    # set up the generic model object.
    gp = CompartmentModel(time=time, curve=curve, aif=aif)
    initial_values = {"F": 50.0, "v": 12.0}

    # fit the model to the curve
    gp.fit_model(initial_values)
    results_std_conv = gp.get_parameters()

    # fit the model to the curve, using fftconvolution
    gp.fit_model(initial_values, fft=True)

    
    results_fft_conv = gp.get_parameters()
    gp.bootstrap(10)
    

    results_bootstrap = gp.get_parameters()
    for p in ['F', 'v', 'MTT', 'Iterations']:
        print 'True value of {}: {}'.format(p, true_values.get(p))
        print 'Fit results std. conv {}: {}'.format(p, results_std_conv.get(p))
        print 'Fit results fft. conv {}: {}'.format(p, results_fft_conv.get(p))
        
    print "Bootstrap estimates: "        
    print [results_bootstrap[x] for x  in ['low estimate', 'mean estimate', 'high estimate']]

    # graphical output
    pylab.plot(gp.time, gp.curve, 'bo')
    pylab.plot(gp.time, gp.fit, 'g-')
    pylab.title("One compartment fit")
    pylab.xlabel("time [s]")
    pylab.ylabel("concentration [a.u.]")
    pylab.legend(('simulated curve', 'model fit'))
    pylab.show()
