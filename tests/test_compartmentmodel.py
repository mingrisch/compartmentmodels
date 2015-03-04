import pytest
import scipy as sp
import numpy as np

import timeit

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
def braindata():
    """ prepare a model instance with startdict, time, aif and synthetic curve +
    additional background noise, ready for fitting
    """
    from compartmentmodels.compartmentmodels import loaddata, savedata
    from compartmentmodels.compartmentmodels import CompartmentModel
    startdict = {'F': 51.0, 'v': 11.2}

    time, aif1, aif2 =loaddata(filename='tests/cerebralartery.csv')    
    # remove baseline signal
    aif = aif1 - aif1[0:5].mean()
    model = CompartmentModel(
        time=time, curve=aif, aif=aif, startdict=startdict)
    # calculate a model curve
    model.curve = model.calc_modelfunction(model._parameters)
    model.curve += 0.02 * aif.max() * np.random.randn(len(time))
    # number of bootstraps
    model.k=100  
    
    return model


@pytest.fixture(scope='module')
def lungdata():
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
    

def test_compartmentModel_has_string_representation(model):
    str_rep=model.__str__()
    
    assert str_rep # an empty string is False, all others are True




def test_compartmentModel_python_convolution(model):
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

    model.time=time
    model.aif=aif

    for i, lam in enumerate(lamdalist):
        curve = inarray[:, i + 2]
        np.testing.assert_array_equal(model.convolution_w_exp(lam), curve,
                                      verbose=False)



def test_compartmentModel_convolution_with_exponential_zero(preparedmodel):
    """ Convolution with an exponential with time constant zero.

    the default implementation will crash here
    """
    int_vector=preparedmodel.intvector()
    conv_w_zero=preparedmodel.convolution_w_exp(0.0)

    np.testing.assert_array_equal(int_vector, conv_w_zero)

def test_compartmentModel_fftconvolution_with_exponential_zero(preparedmodel):
    """ Convolution with an exponential with time constant zero.

    the default implementation will crash here
    """
    int_vector=preparedmodel.intvector()
    conv_w_zero=preparedmodel.convolution_w_exp(0.0, fftconvolution=True)

    np.testing.assert_array_equal(int_vector, conv_w_zero)
    
def do_not_test_compartmentModel_cpython_vs_fft_convolution(model):
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


def do_not_test_compartmentModel_fftconvolution_equal_to_python_convolution(model):
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


def test_compartmentModel_readableParameters_contain_all_keys(preparedmodel):
    assert all([k in preparedmodel.readable_parameters for k in ("F", "v")])

def test_compartmentModel_fit_model_returns_bool(preparedmodel):
    """Test whether the fit routine reports sucess of fitting
    """

    return_value = preparedmodel.fit_model()

    assert (isinstance(return_value, bool))


def test_compartmentModel_start_parameter_conversion(preparedmodel):
    """ are the startparameters converted correctly to raw parameters?

    First, we check manually.
    """

    original_startdict= {'F': 51.0, 'v': 11.2}

    raw_flow=original_startdict.get("F")/6000
    raw_vol=original_startdict.get("v") / 100
    lamda= raw_flow/raw_vol

    par=preparedmodel._parameters
    assert (par[0] == raw_flow) & (par[1] == lamda)

    


def test_compartmentModel_parameter_conversion(preparedmodel):
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


def test_compartmentModel_startdict_is_saved_appropriately(preparedmodel):
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




def test_compartmentModel_fit_model_determines_right_parameters(preparedmodel):
    """ Are the fitted parameters the same as the initial parameters?
    This might become a longer test case...
    """

    start_parameters=preparedmodel._parameters
    return_value = preparedmodel.fit_model()

    assert np.allclose(preparedmodel._parameters, start_parameters)

def test_compartmentModel_fit_model_determines_right_parameters(lungdata):
    """ Are the fitted parameters the same as the initial parameters?
    This might become a longer test case...
    """

    start_parameters=lungdata._parameters
    return_value = lungdata.fit_model()
    print lungdata.OptimizeResult
    assert lungdata._fitted
    #assert np.allclose(lungdata._parameters, start_parameters)


def test_compartmentmodels_bootstrapping_output_dimension_and_type(lungdata):
    """ Is the dimension of the bootstrap_result equal to (2,k) 
    and the dimension of mean.- /std.bootstrap_result equal to (2,)?
    Is the output of type dict?
    Does the output dict contain 7 elements?
    Are 'low estimate', 'mean estimate' and 'high estimate' subdicts in the output dict?
    """
    lungdata.k=100   
    fit_result= lungdata.fit_model()
    bootstrap = lungdata.bootstrap()
    assert (lungdata._bootstrapped == True)
    assert (lungdata.bootstrap_result_raw.shape == (2,100))    
    assert (type(lungdata.readable_parameters) == dict)
    assert (len(lungdata.readable_parameters) == 7)   
    assert ('low estimate' and 'high estimate' and 'mean estimate' in 
            lungdata.readable_parameters)
    assert (type(lungdata.readable_parameters['low estimate']) == dict and 
            type(lungdata.readable_parameters['mean estimate']) == dict and
            type(lungdata.readable_parameters['high estimate']) == dict)


def test_compartmentmodels_bootstrapping_output_content(lungdata):    
    """Is 'low estimate' < 'mean estimate' < 'high estimate'?
    Are fittet Parameters in between 'low estimate' and 'high estimate'?
    """
    lungdata.k=102 
    fit_result= lungdata.fit_model()
    bootstrap = lungdata.bootstrap()
    assert (lungdata.bootstrap_result_raw.shape ==(2, lungdata.k))
    assert (lungdata._bootstrapped == True)
    dict_fit={'F':lungdata.readable_parameters['F'],
                'v':lungdata.readable_parameters['v'],
                'MTT':lungdata.readable_parameters['MTT']
                }
    assert (lungdata.readable_parameters['low estimate'] <
            lungdata.readable_parameters['mean estimate'])
    assert (lungdata.readable_parameters['mean estimate'] <
            lungdata.readable_parameters['high estimate'])
    assert (lungdata.readable_parameters['low estimate'] < dict_fit)
    assert (dict_fit < lungdata.readable_parameters['high estimate'])
 
     
def test_compartmentmodels_bootstrapping_output_content_braindata(braindata):    
    """Is 'low estimate' < 'mean estimate' < 'high estimate'?
    We investigate a cerebral aif here.
    
    Are fitted Parameters in between 'low estimate' and 'high estimate'?
    """
    fit_result= braindata.fit_model()
    bootstrap = braindata.bootstrap()
    assert (braindata._bootstrapped == True)
    dict_fit={'F':braindata.readable_parameters['F'],
                'v':braindata.readable_parameters['v'],
                'MTT':braindata.readable_parameters['MTT']
                }
    assert (braindata.readable_parameters['low estimate'] <
            braindata.readable_parameters['mean estimate'])
    assert (braindata.readable_parameters['mean estimate'] <
            braindata.readable_parameters['high estimate'])
    assert (braindata.readable_parameters['low estimate'] < dict_fit)
    assert (dict_fit < braindata.readable_parameters['high estimate'])
    
def test_compartmentmodel_cython_convolution_equal_to_python(preparedmodel):
    """ Does the cython convolution yield the same result as the python implementation?

    And how much faster ist is, anyway?
    """
    original_curve = preparedmodel.curve

    preparedmodel._use_cython=True

    curve = preparedmodel.calc_modelfunction(preparedmodel._parameters)

    assert np.allequal(originalcurve, curve)
    
    
def test_compartmentmodel_cython_is_faster_than_python(preparedmodel):
    """ Is the cython implementation faster than the python implementation? """

    # to do: how do we get the execution time=?
    preparedmodel_use_cython=False

    pythontime= timeit.timeit(preparedmodel.calc_modelfunction(preparedmodel._parameters), number = 1000)
    # preparedmodel._use_cython=True
    # cythontime= preparedmodel.calc_modelfunction(preparedmodel._parameters)
    print pythontime
    assert False

    # assert cythontime<pythontime

