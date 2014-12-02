import pytest
import scipy as sp
import numpy as np



# set up a fixture for the Compartment model

@pytest.fixture(scope='module')
def model():
    from compartmentmodels.compartmentmodels import CompartmentModel
    model = CompartmentModel()
    return model


@pytest.fixture(scope='module')
def preparedmodel():
    """ prepare a model instance with startdict, time, aif and curve, ready for fitting
    """

    from compartmentmodels.compartmentmodels import CompartmentModel
    startdict = {'F': 51.0, 'v': 11.2}

    time = np.linspace(0, 50, 100)
    aif = np.zeros_like(time)
    aif[(time > 5) & (time < 10)] = 1.0
    model = CompartmentModel(
        time=time, curve=np.zeros_like(time), aif=aif, startdict=startdict)
    # calculate a model curve
    model.curve = model.calc_modelfunction(model._parameters)
    return model
    

@pytest.fixture(scope='module')
def realdata():
    """ prepare a model instance with startdict, time, aif and synthetic curve +
    additional background noise, ready for fitting
    """
    from compartmentmodels.compartmentmodels import loaddata, savedata
    from compartmentmodels.compartmentmodels import CompartmentModel
    startdict = {'F': 51.0, 'v': 11.2}

    t,c,a=loaddata(filename='tests/lung.csv')    
    time = t
    aif = a
    curve = c
    # remove baseline signal
    aif = aif - aif[0:5].mean()
    model = CompartmentModel(
        time=time, curve=curve, aif=aif, startdict=startdict)
    # calculate a model curve
    model.curve = model.calc_modelfunction(model._parameters)
    model.curve += 0.02 * aif.max() * np.random.randn(len(time))
    return model

@pytest.fixture(scope='module')
def realcurve():
    """ prepare a model instance with startdict, time, aif and real curve ready for
    fitting
    """
    from compartmentmodels.compartmentmodels import loaddata, savedata
    from compartmentmodels.compartmentmodels import CompartmentModel
    startdict = {'F': 51.0, 'v': 11.2}

    t,c,a=loaddata(filename='tests/lung.csv')    
    time = t
    aif = a
    # remove baseline signal
    curve = c - c[0:5].mean()
    model = CompartmentModel(
        time=time, curve=curve, aif=aif, startdict=startdict)
    #model.curve += 0.002 * curve.max() * np.random.randn(len(time))
    return model
    

def test_genericModel_has_string_representation(model):
    str_rep=model.__str__()
    
    assert str_rep # an empty string is False, all others are True



def test_genericModel_set_and_get_time(model):
    time = sp.linspace(0, 20, 50)
    model.set_time(time)
    np.testing.assert_array_equal(model.get_time(), time, verbose=False)


def test_genericModel_set_and_get_aif(model):
    aif = np.random.randn(50)
    model.set_aif(aif)
    np.testing.assert_array_equal(model.get_aif(), aif, verbose=False)


def test_genericModel_set_and_get_curve(model):
    curve = np.random.rand(50)
    model.set_curve(curve)
    np.testing.assert_array_equal(model.get_curve(), curve, verbose=False)




def test_genericModel_python_convolution(model):
    # load a curve that was calculated with pmi
    # 'aif' is a boxcar function;
    testfile = 'tests/convolutions.csv'
    with open(testfile) as f:
        header = f.readline()
    header = header.lstrip('# ').rstrip()

    header = header.split(',')
    lamdalist = [np.float(el) for el in (header[2:])]

    inarray = np.loadtxt(testfile)
    time = inarray[:, 0]
    aif = inarray[:, 1]

    model.set_time(time)
    model.set_aif(aif)

    for i, lam in enumerate(lamdalist):
        curve = inarray[:, i + 2]
        np.testing.assert_array_equal(model.convolution_w_exp(lam), curve,
                                      verbose=False)



def test_genericModel_convolution_with_exponential_zero(preparedmodel):
    """ Convolution with an exponential with time constant zero.

    the default implementation will crash here
    """
    int_vector=preparedmodel.intvector()
    conv_w_zero=preparedmodel.convolution_w_exp(0.0)

    np.testing.assert_array_equal(int_vector, conv_w_zero)

def test_genericModel_fftconvolution_with_exponential_zero(preparedmodel):
    """ Convolution with an exponential with time constant zero.

    the default implementation will crash here
    """
    int_vector=preparedmodel.intvector()
    conv_w_zero=preparedmodel.convolution_w_exp(0.0, fftconvolution=True)

    np.testing.assert_array_equal(int_vector, conv_w_zero)
    
def do_not_test_genericModel_cpython_vs_fft_convolution(model):
    """ TEst whether a fft convolution yields the same result as the cpython
    implementation of the discrete convolution

    to do: this is currently not tested - we need some research first (issue #17)
    """
    testfile = 'tests/convolutions.csv'

    with open(testfile) as f:
        header = f.readline()
    header = header.lstrip('# ').rstrip()

    header = header.split(',')
    lamdalist = [np.float(el) for el in (header[2:])]

    inarray = np.loadtxt(testfile)
    time = inarray[:, 0]
    aif = inarray[:, 1]

    model.set_time(time)
    model.set_aif(aif)

    for i, lam in enumerate(lamdalist):
        # this curve was calcualted with the cpython convolution
        curve = inarray[:, i + 2]

        np.testing.assert_array_equal(model.convolution_w_exp(lam, fftconvolution=True),
        curve,verbose=False)


def do_not_test_genericModel_fftconvolution_equal_to_python_convolution(model):
    """ Test whether sp.fftconvolve yields a similar result to the discrete
    convolution
    to do: this is currently not tested - we need some research first (issue #17)
    """

    time = np.linspace(0, 50, 2000)
    aif = np.zeros_like(time)
    aif[(time > 5) & (time < 15)] = 1

    model.set_time(time)
    model.set_aif(aif)

    lamdalist = [4.2, 3.9, 0.1]

    for i, lam in enumerate(lamdalist):

        np.testing.assert_array_equal(model.convolution_w_exp(lam, fftconvolution=True),
        model.convolution_w_exp(lam, fftconvolution=False),verbose=False)


def test_genericModel_readableParameters_contain_all_keys(preparedmodel):
    assert all([k in preparedmodel.readable_parameters for k in ("F", "v")])

def test_genericModel_fit_model_returns_bool(preparedmodel):
    """Test whether the fit routine reports sucess of fitting
    """

    return_value = preparedmodel.fit_model()

    assert (isinstance(return_value, bool))


def test_genericModel_start_parameter_conversion(preparedmodel):
    """ are the startparameters converted correctly to raw parameters?

    First, we check manually.
    """

    original_startdict= {'F': 51.0, 'v': 11.2}

    raw_flow=original_startdict.get("F")/6000
    raw_vol=original_startdict.get("v") / 100
    lamda= raw_flow/raw_vol

    par=preparedmodel._parameters
    assert (par[0] == raw_flow) & (par[1] == lamda)

    


def test_genericModel_parameter_conversion(preparedmodel):
    """ check the conversions from physiological to raw, and back
    """
    
    original_startdict= {'F': 51.0, 'v': 11.2}
    raw_par= preparedmodel._parameters
    
    readable_dict= preparedmodel.get_parameters()
    # test whether the dictionaries contain i) the same keys and ii) the corresponding values are equal. All keys from the start dict have to be in the output dict. Additionally, the readable_dict may contain additional keys, which are not checked.
    for key, value in original_startdict.iteritems():
        if key in readable_dict:
            assert np.allclose(original_startdict.get(key) , readable_dict.get(key))
        else:
            assert False, "Key {} is not contained in readable dictionary".format(key)


def test_genericModel_startdict_is_saved_appropriately(preparedmodel):
    """ is the startdict from constructor saved correctly?
    """

    original_startdict = {'F': 51.0, 'v': 11.2}
    readable_dict=preparedmodel.readable_parameters
    # we need to check whether all original keys/values are contained in the model parameter dict:
    for key, value in original_startdict.iteritems():
        if key in readable_dict:
            assert np.allclose(original_startdict.get(key), readable_dict.get(key))
        else:
            assert False,  "Key {} is not contained in readable dictionary".format(key)




def test_genericModel_fit_model_determines_right_parameters(preparedmodel):
    """ Are the fitted parameters the same as the initial parameters?
    This might become a longer test case...
    """

    start_parameters=preparedmodel._parameters
    return_value = preparedmodel.fit_model()

    assert np.allclose(preparedmodel._parameters, start_parameters)

def test_genericModel_fit_model_determines_right_parameters(realdata):
    """ Are the fitted parameters the same as the initial parameters?
    This might become a longer test case...
    """

    start_parameters=realdata._parameters
    return_value = realdata.fit_model()
    print realdata.OptimizeResult
    assert realdata._fitted
    #assert np.allclose(realdata._parameters, start_parameters)


def test_compartmentmodels_bootstrapping_output_dimension_and_type(realdata):
    """ Is the dimension of the bootstrap_result equal to (2,k) 
    and the dimension of mean.- /std.bootstrap_result equal to (2,)?
    Is the output of type dict?
    Does the output dict contain 7 elements?
    Are 'low estimate', 'mean estimate' and 'high estimate' subdicts in the output dict?
    """
    realdata.k=100   
    fit_result= realdata.fit_model()
    bootstrap = realdata.bootstrap()
    assert (realdata._bootstrapped == True)
    assert (realdata.bootstrap_result_raw.shape == (2,100))    
    assert (type(realdata.readable_parameters) == dict)
    assert (len(realdata.readable_parameters) == 7)   
    assert ('low estimate' and 'high estimate' and 'mean estimate' in 
            realdata.readable_parameters)
    assert (type(realdata.readable_parameters['low estimate']) == dict and 
            type(realdata.readable_parameters['mean estimate']) == dict and
            type(realdata.readable_parameters['high estimate']) == dict)
    
     
def test_compartmentmodels_bootstrapping_output_content(realdata):    
    """Is 'low estimate' < 'mean estimate' < 'high estimate'?
    Are fittet Parameters in between 'low estimate' and 'high estimate'?
    """
    realdata.k=1000  
    fit_result= realdata.fit_model()
    bootstrap = realdata.bootstrap()
    assert (realdata._bootstrapped == True)
    dict_fit={'F':realdata.readable_parameters['F'],
                'v':realdata.readable_parameters['v'],
                'MTT':realdata.readable_parameters['MTT']
                }
    assert (realdata.readable_parameters['low estimate'] <
            realdata.readable_parameters['mean estimate'])
    assert (realdata.readable_parameters['mean estimate'] <
            realdata.readable_parameters['high estimate'])
    assert (realdata.readable_parameters['low estimate'] < dict_fit)
    assert (dict_fit < realdata.readable_parameters['high estimate'])
            

def test_compartmentmodels_bootstrapping_output_dimension_and_type2(realcurve):
    """ Is the dimension of the bootstrap_result equal to (2,k) 
    and the dimension of mean.- /std.bootstrap_result equal to (2,)?
    Is the output of type dict?
    Does the output dict contain 7 elements?
    Are 'low estimate', 'mean estimate' and 'high estimate' subdicts in the output dict?
    """
    realcurve.k=100   
    fit_result= realcurve.fit_model()
    bootstrap = realcurve.bootstrap()
    assert (realcurve._bootstrapped == True)
    assert (realcurve.bootstrap_result_raw.shape == (2,100))    
    assert (type(realcurve.readable_parameters) == dict)
    assert (len(realcurve.readable_parameters) == 7)   
    assert ('low estimate' and 'high estimate' and 'mean estimate' in 
            realcurve.readable_parameters)
    assert (type(realcurve.readable_parameters['low estimate']) == dict and 
            type(realcurve.readable_parameters['mean estimate']) == dict and
            type(realcurve.readable_parameters['high estimate']) == dict)    
    
    
def test_compartmentmodels_bootstrapping_output_content2(realcurve):    
    """Is 'low estimate' < 'mean estimate' < 'high estimate'?
    Are fittet Parameters in between 'low estimate' and 'high estimate'?
    """
    realcurve.k=1000  
    fit_result= realcurve.fit_model()
    bootstrap = realcurve.bootstrap()
    assert (realcurve._bootstrapped == True)
    dict_fit={'F':realcurve.readable_parameters['F'],
                'v':realcurve.readable_parameters['v'],
                'MTT':realcurve.readable_parameters['MTT']
                }
    assert (realcurve.readable_parameters['low estimate'] <
            realcurve.readable_parameters['mean estimate'])
    assert (realcurve.readable_parameters['mean estimate'] <
            realcurve.readable_parameters['high estimate'])
    assert (realcurve.readable_parameters['low estimate'] < dict_fit)
    assert (dict_fit < realcurve.readable_parameters['high estimate'])    
    
