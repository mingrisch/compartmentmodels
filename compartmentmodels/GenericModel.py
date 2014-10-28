# cython: profile=True
import scipy as sp
from scipy import optimize

#import pyximport

# pyximport.install()
#from cy_conv_exp import cy_conv_exp


class GenericModel:

    """This is a base class for all models. It provides the general
    framework and contains a fit function for a simple one compartment
    model. For other models, we recommend subclassing this class and
    overwriting/redefining the function calc_residuals(self,
    parameters)"""

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
        self.parameters = []

        # needed for calculation of the Akaike information criterion:
        self.nooffreeparameters = 2

        # this dictionary will contain human-readable (parameter,value) entries
        self.readable_parameters = {'FP': 0.0,
                                    'VP': 0.0,
                                    'TP': 0.0,
                                    'E': 0.0,
                                    'PS': 0.0,
                                    'VE': 0.0,
                                    'TE': 0.0}

    def __str__(self):
        return "Generic model"

    # get functions
    def get_parameters(self):
        """To be used after a successful fit.
        returns a dictionary with keys FP, VP, TP, PS, VE, TE
        """
        FP = self.parameters[0] * 6000.
        VP = self.parameters[0] / self.parameters[1] * 100
        TP = 1 / self.parameters[1]
        self.readable_parameters["FP"] = FP
        self.readable_parameters["VP"] = VP
        self.readable_parameters["TP"] = TP
        self.readable_parameters["E"] = None
        self.readable_parameters["PS"] = None
        self.readable_parameters["VE"] = None
        self.readable_parameters["TE"] = None

        return self.readable_parameters

    def get_raw_parameters(self):
        return self.parameters

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
        self.parameters = newparameters

    def set_time(self, newtime):
        self.time = newtime

    def set_curve(self, newcurve):
        self.curve = newcurve

    def set_aif(self, newaif):
        self.aif = newaif

    # convolution of aif with an exponential
    def convolution_w_exp(self, lamda):
        """ returns the convolution of self.aif with an exponential
         exp(-lamda*t). we follow the notation introduced in
        http://edoc.ub.uni-muenchen.de/14951/
        """
#        y=cy_conv_exp(list(self.time),list(self.curve),list(self.aif),tau)
        y = cy_conv_exp(self.time, self.curve, self.aif, lamda)
        return sp.asarray(y)

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

    def calc_residuals(self, parameters, fjac=None):
        """This function calculates the residuals for a
        one-compartment model with residual function
        p[0]*exp(-t*p[1]).  self.residuals is set to the resulting
        array, furthermore, the sum of resulting array is returned.  Note:
        This function will be called from the solver quite often. For
        optimizing performance, someone could rewrite it in c or
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
        """this function attempts to fit the model to the curve, using
        the startparameters as initial value.  on return, self.rc
        contains the return parameter of the leastsq routine, and
        self.parameters is set to the parameter estimates, self. fit
        is set to the estimated model curve.  after fitting, the
        method get_aic() is available to calculate the Akaike
        information criterion."""
        startparameters = self.convert_startdict(startdict)

        # set up the parinfo-list of dictionaries. this is necessary if
        # we wish to constrain all parameters to positive values.
        npar = len(startparameters)
        parinfo = [{'value': startparameters[i],
                    'limited':[constrained, False],
                    'limits':[1.0e-7, 1]}
                   for i in range(npar)]

        # we need an additional constraint for the temporal parameters
        mp = mpfit(
            self.calc_residuals, startparameters, parinfo=parinfo, quiet=1)
        # print "Status :%d" % mp.status
        # print "Params :", mp.params
        # print mp.errmsg

        self.parameters = mp.params
        # print "Fitting results:" ,self.parameters
        # print "Other stuff: ", fit_results
        # print "Status message: ", mesg
        # print "Return code : ", ier
        self.rc = mp.status
        self.fit = self.curve - self.calc_residuals(self.parameters)[1]

        return

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
