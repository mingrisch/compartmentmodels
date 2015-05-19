# -*- coding: utf-8 -*-

import numpy as np
# import scipy as sp
from scipy import stats
from scipy.optimize import minimize

# cython:
try:
    import pyximport
    pyximport.install()

    import c_convolution_exp
    cython_available = True

except:
    # cython import has not worked:
    cython_available = False
    print "Cython is apparently not available. Continuing without..."

# helper functions for saving and loading 'model datasets'


def loaddata(filename, separator=',', comment='#'):
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
        t, c, a = np.loadtxt(filename, comments=comment,
                             delimiter=separator, unpack=True)
    except:
        raise IOError

    # sanity checks:
    if (
            (len(t) != len(a)) or
            (len(t) != len(c)) or
            (len(a) != len(c))):
        raise ValueError

    return (t, c, a)


def savedata(filename, time, curve, aif):
    """This function saves three 1D arrays time, curve and AIF
    into a file that complies to the format accepted by loaddata"""
    try:
        np.savetxt(filename, np.transpose((time, curve, aif)), delimiter=',')
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

    _fitparameters: np.ndarray
        array of raw parameters, used internally for the calculation of model
        fits

    _use_cython: bool
        do we have cython, or use it?


    _fitted: bool
        has a fit been performed?

    phys_parameters: dict
        Dictionary of physiological parameters. These can be used as start
        parameters for the fit, and are calculated from self._fitparameters after
        a successfull fit




    """

    def __init__(self, time=np.empty(1), curve=np.empty(1), aif=np.empty(1),
                 startdict={'F': 50.0, 'v': 12.2}, use_cython=cython_available):
        # todo: typecheck for time, curve, aif
        # they should be numpy.ndarrays with the same size and dtype float
        self.time = time
        self.aif = aif
        self.curve = curve
        self.residuals = curve
        self.fit = np.zeros_like(curve)
        self.aic = 0.0
        self._fitparameters = np.zeros(2)

        self.k = 500
        self._use_cython = use_cython
        self._fitted = False
        self._bootstrapped = False

        # Perform convolution using fft or linear interpolation?
        self.fft = False
        # this dictionary will contain human-readable (parameter,value) entries
        self.phys_parameters = startdict

        # this will store the OptimizeResult object
        self.OptimizeResult = None

        # convert the dictionary entry to 'raw' parameters:
        self._fitparameters = self._phys_to_fit(startdict)

        # defy method and bounds, which fit_model will use:
        self.constrained = False
        self.set_constraints(self.constrained)

    def __str__(self):
        return "Base class for compartment models."

    # get functions
    def _fit_to_phys(self, aslist=False):
        """Convert the fitted parameters back to physiological parameters.

        To be used after a successful fit.
        Converts the 'raw' fit parameters to the 'physiological' model
        parameters and saves them in self.phys_parameters.
        Parameters
        ----------
        aslist : bool
            if True, return the values as list, if False, return the values as dictionary (as previously)

        Returns
        -------
        None


        Notes
        -----

        For the one-compartment model, these parameters are a flow, a volume
        and the corresponding transit time. Derived models will likely have to
        override this method.  """

        F = self._fitparameters[0] * 6000.
        v = self._fitparameters[0] / self._fitparameters[1] * 100
        mtt = 1 / self._fitparameters[1]

        if aslist:
            # this is needed e.g. by self.bootstrap
            return [F, v, mtt]

        self.phys_parameters["F"] = F
        self.phys_parameters["v"] = v
        self.phys_parameters["MTT"] = mtt

        if self._fitted:
            self.phys_parameters["Iterations"] = self.OptimizeResult.nit

        if self._bootstrapped:
            self.phys_parameters["low estimate"] = {'F': self.bootstrap_percentile[
                0, 0], 'v': self.bootstrap_percentile[0, 1], 'MTT': self.bootstrap_percentile[0, 2]}
            self.phys_parameters["mean estimate"] = {'F': self.bootstrap_percentile[
                1, 0], 'v': self.bootstrap_percentile[1, 1], 'MTT': self.bootstrap_percentile[1, 2]}
            self.phys_parameters["high estimate"] = {'F': self.bootstrap_percentile[
                2, 0], 'v': self.bootstrap_percentile[2, 1], 'MTT': self.bootstrap_percentile[2, 2]}

        return

    # convolution of aif with an exponential
    def convolution_w_exp(self, lamda, fftconvolution=False):
        """ Convolution of self.aif with an exponential.

        This function returns the convolution of self.aif with an exponential
        exp(-lamda*t). Currently, two implementations are available: per
        default, we calculate the convolution analytically with a linearly
        interpolated AIF, following the notation introduced in
        http://edoc.ub.uni-muenchen.de/14951/. 

        Alternatively, convolution via fft can be used. Currently, the tests
        for the FFT convolution that compare FFT to standard convolution fail.
        Presumably, the reason is the difference between sinc-interpolation (by
        FFT) and linear interpolation. 

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

        elif self._use_cython:
            # calculcate the discrete  convolution with the cpython
            # implementation
            return c_convolution_exp.c_convolution_exp(self.time, self.aif, lamda)
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
        # return np.asarray(y)

    def intvector(self):
        """This function calculates the convolution of the arterial
        input function with a constant. Basically, this is an
        integration of the AIF."""
        y = self.aif
        x = self.time
        # no need for a for-loop

        ysum = y[1:] + y[0:-1]
        # dx=x[1:]-x[0:-1]
        dx = np.diff(x)
        integral = np.zeros_like(x)
        integral[1:] = np.cumsum(ysum * dx / 2)

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

    def _phys_to_fit(self, startdict):
        """
        Take a dictionary containing start values for Fp,vp,PS,ve and
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
        Fp = startdict.get("F") / 6000.  # we need to convert to SI
        vp = startdict.get("v") / 100
        lamda = 1. / (vp / Fp)

        # store the parameters in self._fitparameters:
        self._fitparameters = np.asarray([Fp, lamda])
        return np.asarray([Fp, lamda])

    def set_constraints(self, constrained):
        """ This function sets the contraints for fitting. 

        Do we really ned this function? I mean, we always use positivity
        constraints, don't we?  """

        self.constrained = constrained

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
        # convert start dict to self._fitparameters
        if startdict is None:
            startparameters = self._fitparameters
        else:
            startparameters = self._phys_to_fit(startdict)

        self.set_constraints(constrained)

        fit_results = minimize(self._calc_residuals, startparameters, args=(self.curve,),
                               method=self.method,
                               bounds=self.bounds)

        self._fitparameters = fit_results.x

        # store the Optimize Result, in case we need it later on
        self.OptimizeResult = fit_results
        self.fit = self.calc_modelfunction(self._fitparameters)
        self._fitted = fit_results.success
        # convert the fit parameters back to physiological parameters
        self._fit_to_phys()

        return fit_results.success

    def get_AIC(self):
        """this method returns the corrected Akaike information
        criterion (Glatting 07).  It is only available after a
        successful fit"""
        if self._fitted:
            n = self.fit.size
            npar = len(self._fitparameters)
            ss = np.sum(np.square(self.residuals))

            aic = n * np.log(ss / n) + 2 * (npar + 1) + 2 * (npar + 1) * (npar + 2) / (n - npar - 2)
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
        """

        if not self._fitted:
            raise ValueError
        if k:
            self.k = k

        # need to change variables for bootstrapping
        original_curve = self.curve
        original_fit = self.fit
        original_parameters = self._fitparameters
        original_phys_parameters = self.phys_parameters

        # set of residuals calculated for bootstrapping
        residuals_bootstrap = (self.curve - self.fit)

        # shapiro-wilk test for normaly distributed residuals
        w, p = stats.shapiro(residuals_bootstrap)
        # print 'test statistic:', w, 'p-value:', p
        # if p < 0.05:
        #     raise ValueError(
        #         'probably not normal distributed residuals. Try another model')

        # array, which will be overwritten with the results
        #:self.bootstrap_result_raw = np.zeros((len(self._fitparameters), self.k))
        # here we store converted parameters that are returned by
        # _fit_to_phys(aslist=True)
        self.bootstrap_result = np.zeros(
            (len(self._fit_to_phys(aslist=True)), self.k))

        # todo: self.bootstrap_result should contain the physiological
        # parameters, derived from the fit parameters by self._fit_to_phys, or
        # whatever.

        # bootstrapping loop
        for i in range(self.k):
            sample_index = np.random.randint(
                0, residuals_bootstrap.shape[0], residuals_bootstrap.shape)
            self.curve = original_fit + residuals_bootstrap[sample_index]
            self.fit_model(original_phys_parameters)
            # self.bootstrap_result_raw[:, i] = self._fitparameters
            self.bootstrap_result[:, i] = self._fit_to_phys(aslist=True)
            # todo: here, we need the conversion to physiological parameters

        # restore variables
        self.curve = original_curve
        self.fit = original_fit
        self._fitparameters = original_parameters
        self.phys_parameters = original_phys_parameters
        self._bootstrapped = True

        self.bootstrap_percentile = np.percentile(
            self.bootstrap_result, [17, 50, 83], axis=1)

        return self._fit_to_phys()
        return


class TwoCXModel(CompartmentModel):

    """
    The two compartment exchange model


    this needs to be implemented for the whole script. Taken from S P Sourbron and D L Buckley - tracer kinetic modeliling in MRI
    T_c = v_p / F_p    
    T_e = v_e / PS    mean transit time of EES
    T = (v_p + v_e) / F_p     mean transit time of combined system
    sigma_+ = ((T+T_e) + sqrt(  (T+T_e)**2   -  4*T_c*T_e  ))   /   (2*T_c*T_e)
    sigma_- = ((T+T_e) - sqrt(  (T+T_e)**2   -  4*T_c*T_e  ))   /   (2*T_c*T_e)    
    R(t) = ((T*sigma_+   -  1)*sigma_-  * exp(-t*sigma_-)  +  (1-T*sigma_-)*sigma_+  * exp(-t*sgima_+)) / (sigma_+  - sigma_-)
    R(s) = (T+ s*T_c*T_e)/(s**2 * T_c * T_e + s*(T+T_e)+1)

    """

    def __init__(self, time=np.empty(1), curve=np.empty(1), aif=np.empty(1), startdict={'Fp': 51.0, 'vp': 11.2, 'PS': 4.9, 've': 13.2}):
        CompartmentModel.__init__(self, time, curve, aif, startdict)
        # override the value of the compartment model:
        self.startdict = startdict
        self.nooffreeparameters = 4

    def __str__(self):
        return "2CX model"

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

        # we need to make sure how these parameters are defined. kp and
        # km allow a discrimination between the two exponential terms,
        # since kp is always larger than km

        # with the internal fit parameters
        # [ Fp, delta, E, Km], with delta = kp-km, we can write the model curve as

        # R(t)  = Fp * (E * exp(-tKm) + (1-E) exp(-t(Km+delta)
        p = parameters
        modelcurve = p[0] * (
            (p[2] * self.convolution_w_exp(p[3], fftconvolution=self.fft)) +
            (1 - p[2]) *
            self.convolution_w_exp(p[3] + p[1], fftconvolution=self.fft)
        )

        return modelcurve

    def _phys_to_fit(self, startdict):
        """
        Convert the physiological parameters to parameters which are suitable for fitting..

        Here we use the notation used in the PhD thesis of M Ingrisch, which is slightly clearer than the PMB notation.

        """

        Fp = startdict.get("Fp") / 6000.
        vp = startdict.get("vp") / 100.
        PS = startdict.get("PS") / 6000.
        ve = startdict.get("ve") / 100.

        TB = vp / Fp
        TE = ve / PS
        TP = vp / (PS + Fp)

        KP = 0.5 * \
            (1. / TP + 1. / TE +
             np.sqrt(np.square(1. / TP + 1. / TE) - 4. / TE / TB))
        KM = 0.5 * \
            (1. / TP + 1. / TE -
             np.sqrt(np.square(1. / TP + 1. / TE) - 4. / TE / TB))
        E = (KP - 1 / TB) / (KP - KM)

        delta = KP - KM
        # parameters=[Fp,delta,E,KM]
        return [Fp, delta, E, KM]

    def set_constraints(self, constrained):
        """ This function sets the contraints for fitting. 
        """

        self.constrained = constrained

        # somethings wrong with the bounds. get error when testing, like:
        # TypeError: unsupported operand type(s) for -: 'list' and 'list'
        if constrained:
            self.bounds = [(0, None), (0, None), (0, None), (0, None)]
            self.method = 'L-BFGS-B'
        else:
            self.bounds = [
                (None, None), (None, None), (None, None), (None, None)]
            self.method = 'BFGS'

    def _fit_to_phys(self, aslist=False):
        """ Convert the fitted parameters back to physiological parameters.

        Again we use the notation from the PhD thesis of M Ingrisch. 


        Parameters
        ---------
        aslist : bool
            if True, return the parameters as list with values Fp, vp, TP, PS, ve, TE. Otherwise, set the dictionary self.phys_parametrs.

        Returns : 
        --------
         a list, when aslist=True, nothing if aslist=False



        """
        # self._fitparameters=[Fp, delta, E, KM]

        delta = self._fitparameters[1]
        KM = self._fitparameters[3]
        KP = KM + delta
        EM = self._fitparameters[2]
        Fp = self._fitparameters[0]

        TB = 1.0 / (KP - EM * (KP - KM))
        TE = 1.0 / (TB * KP * KM)
        TP = 1.0 / (KP + KM - 1.0 / TE)

        PS = Fp * (TB / TP - 1)
        ve = PS * TE
        E = PS / (PS + Fp)

        if aslist:
            return [Fp, vp, TP, PS, ve, TE]

        self.phys_parameters["Fp"] = Fp * 6000.0
        self.phys_parameters["vp"] = Fp * TB * 100.0
        self.phys_parameters["TP"] = TP
        self.phys_parameters["E"] = E
        self.phys_parameters["PS"] = PS * 6000.0
        self.phys_parameters["ve"] = ve * 100.0
        self.phys_parameters["TE"] = TE

        if self._fitted:
            self.phys_parameters["Iterations"] = self.OptimizeResult.nit

        if self._bootstrapped:
            # store the bootstrap percentiles into self.phys_parametes
            result_name_list = [
                'low estimate', 'mean estimate', 'high estimate']

            for estimate in result_name_list:
                self.phys_parameters[estimate] = {
                    'Fp': self.bootstrap_percentile[j, 0],
                    'vp': self.bootstrap_percentile[j, 1],
                    'TP': self.bootstrap_percentile[j, 2], 
                    'E': self.bootstrap_percentile[j, 3],
                    'PS': self.bootstrap_percentile[j, 4], 
                    've': self.bootstrap_percentile[j, 5],
                    'TE': self.bootstrap_percentile[j, 6]
                                                        }

        return


class TwoCUModel(CompartmentModel):

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

    phys_parameters: dict
        Dictionary of physiological parameters. These can be used as start
        parameters for the fit, and are calculated from self._fitparameters after
        a successfull fit


    """

    def __init__(self, time=np.empty(1), curve=np.empty(1), aif=np.empty(1),
                 startdict={'Fp': 51.0, 'v': 11.2, 'PS': 4.9}):
        CompartmentModel.__init__(self, time, curve, aif, startdict)
        # override the value of the compartment model:
        self.startdict = startdict
        self.nooffreeparameters = 3

    def __str__(self):
        return "2CU model"

    def calc_modelfunction(self, parameters):
        """This function calculates the residuals for a two
        compartment uptake model with the residual function
        p[0]*((1-p[2])*exp[-t*p[1])+ p[2]), or, correspondingly,
        Fp*(exp(-t/TP)+ E(1-exp(-t/TP)))=
        Fp*((1-E)*exp(-t/TP)+E)"""

        p = parameters
        modelcurve = p[0] * ((1 - p[2]) * self.convolution_w_exp(p[1]) +
                             p[2] * self.intvector())

        return modelcurve

    def _phys_to_fit(self, startdict):
        Fp = startdict.get("Fp") / 6000.
        vp = startdict.get("vp") / 100.
        PS = startdict.get("PS") / 6000.
        # ve=startdict.get("ve")/100.

        E = PS / (PS + Fp)
        TP = vp / (PS + Fp)

        return [Fp, 1. / TP, E]

    def set_constraints(self, constrained):
        """ This function sets the contraints for fitting. 
        """

        self.constrained = constrained

        if constrained:
            self.bounds = [(0, None), (0, None), (0, None)]
            self.method = 'L-BFGS-B'
        else:
            self.bounds = [
                (None, None), (None, None), (None, None)]
            self.method = 'BFGS'

    def _fit_to_phys(self, aslist=False):
        """To be used after a successful fit.
        converts fitted parameters to physiological parameters. Results are stored in  dictionary self.phys_parameters with keys Fp, vp, TP, PS, ve, TE
        conversion into physiological parameters follows sourbron, mrm09
        """
        Fp = self._fitparameters[0]
        TP = 1. / self._fitparameters[1]
        E = self._fitparameters[2]
        vp = TP * Fp / (1.0 - E)
        PS = E * Fp / (1.0 - E)

        self.phys_parameters["Fp"] = Fp * 6000
        self.phys_parameters["vp"] = vp * 100
        self.phys_parameters["TP"] = TP
        self.phys_parameters["PS"] = PS * 6000
        self.phys_parameters["E"] = E
        # self.phys_parameters["ve"]=None
        # self.phys_parameters["TE"]=None
        
        if aslist:
            return[Fp,vp,TP,PS,E]

        if self._fitted:
            self.phys_parameters["Iterations"] = self.OptimizeResult.nit

        if self._bootstrapped:
            # convert bootstrapped_raw to boostrapped_physiological
            # self.bootstrap_result_physiological = np.zeros((5, self.k))
            # assert self.bootstrap_result_raw.shape[1] == self.k
            # for i in range(self.k):
            #     Fp_bootstrap = self.bootstrap_result_raw[0, i]
            #     TP_bootstrap = 1 / self.bootstrap_result_raw[1, i]
            #     E_bootstrap = self.bootstrap_result_raw[2, i]
            #     vp_bootstrap = TP_bootstrap * \
            #         Fp_bootstrap / (1.0 - E_bootstrap)
            #     PS_bootstrap = E_bootstrap * Fp_bootstrap / (1.0 - E_bootstrap)

            #     Fp_bootstrap = Fp_bootstrap * 6000
            #     vp_bootstrap = vp_bootstrap * 100
            #     PS_bootstrap = PS_bootstrap * 6000

            #     self.bootstrap_result_physiological[:, i] = (
            #         Fp_bootstrap, vp_bootstrap, TP_bootstrap, E_bootstrap, PS_bootstrap)

            # self.bootstrap_percentile = np.percentile(
            #     self.bootstrap_result_physiological, [17, 50, 83], axis=1)

            result_name_list = [
                'low estimate', 'mean estimate', 'high estimate']
            for estimate in result_name_list:
                self.phys_parameters[estimate] = {'Fp': self.bootstrap_percentile[j, 0], 'vp': self.bootstrap_percentile[j, 1],
                                                                    'TP': self.bootstrap_percentile[j, 2], 'E': self.bootstrap_percentile[j, 3],
                                                                    'PS': self.bootstrap_percentile[j, 4]
                                                                    }

        return self.phys_parameters


if __name__ == '__main__':
    import pylab
    npt = 200
    tmax = 50

    # inputs. one could, in principle, load this dataset.
    # but we don't have one yet.

    time = np.linspace(0, tmax, num=npt)
    aif = np.square(time) * np.exp(-time / 2)
    curve = aif * 0.0
    # aif=aif.astype('float64')
    # curve=curve.astype('float64')
    gptmp = CompartmentModel(time, curve, aif)

    # generate an aif with recirculation:
    recirculation1 = gptmp.convolution_w_exp(1. / 8.) / 8
    recirculation1 = np.roll(recirculation1, 10 * 4)
    recirculation1[0:10 * 4] = 0
    recirculation2 = gptmp.convolution_w_exp(100.) / 100
    recirculation2 = np.roll(recirculation2, 20 * 4)
    recirculation2[0:20 * 4] = 0
    aif = aif + recirculation1 + recirculation2
    gptmp.aif = aif

    true_values = {"F": 45.0, "v": 10.0}
    true_parameters = gptmp._phys_to_fit(true_values)
    curve = true_parameters[0] * gptmp.convolution_w_exp(true_parameters[1])
    max_curve = np.amax(curve)
    # to do: what the hell is this:
    noise = 0.08 * max_curve * (np.random.randn(len(curve)))
    curve = curve + noise
# ____ end of input calculations

    # set up the generic model object.
    gp = CompartmentModel(time=time, curve=curve, aif=aif)
    initial_values = {"F": 50.0, "v": 12.0}

    # fit the model to the curve
    gp.fit_model(initial_values)
    gp._fit_to_phys()

    # fit the model to the curve, using fftconvolution
    gp.fit_model(initial_values, fft=True)

    results_fft_conv = gp._fit_to_phys()
    gp.bootstrap(10)

    results_bootstrap = gp._fit_to_phys()
    for p in ['F', 'v', 'MTT', 'Iterations']:
        print 'True value of {}: {}'.format(p, true_values.get(p))
        print 'Fit results std. conv {}: {}'.format(p, results_std_conv.get(p))
        print 'Fit results fft. conv {}: {}'.format(p, results_fft_conv.get(p))

    print "Bootstrap estimates: "
    print 'low estimate: ', results_bootstrap['low estimate']
    print 'mean estimate:', results_bootstrap['mean estimate']
    print 'high estimate:', results_bootstrap['high estimate']

    # graphical output
    pylab.plot(gp.time, gp.curve, 'bo')
    pylab.plot(gp.time, gp.fit, 'g-')
    pylab.title("One compartment fit")
    pylab.xlabel("time [s]")
    pylab.ylabel("concentration [a.u.]")
    pylab.legend(('simulated curve', 'model fit'))
    pylab.show()
