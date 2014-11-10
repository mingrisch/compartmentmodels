import pytest
import scipy as sp
import numpy as np



# set up a fixture for the Generic model

@pytest.fixture(scope='module')
def model():
    from compartmentmodels.GenericModel import GenericModel
    model = GenericModel()
    return model


@pytest.fixture(scope='module')
def preparedmodel():
    """ prepare a model instance with startdict, time, aif and curve, ready for fitting
    """

    from compartmentmodels.GenericModel import GenericModel
    startdict = {'F': 51.0, 'v': 11.2}

    time = np.linspace(0, 50, 100)
    aif = np.zeros_like(time)
    aif[(time > 5) & (time < 10)] = 1.0
    model = GenericModel(
        time=time, curve=np.zeros_like(time), aif=aif, startdict=startdict)
    # calculate a model curve
    model.curve = model.calc_modelfunction(model._parameters)

    return model


def test_genericModel_has_string_representation(model):
    assert model.__str__() == "Generic model"


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


def test_genericModel_cpython_vs_fft_convolution(model):
    """ TEst whether a fft convolution yields the same result as the cpython
    implementation of the discrete convolution

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

        np.testing.assert_array_equal(model.convolution_w_exp(lam,
                                                              fftconvolution=True), curve, verbose=False)


def test_genericModel_fftconvolution_equal_to_python_convolution(model):
    """ Test whether sp.fftconvolve yields a similar result to the discrete
    convolution"""

    time = np.linspace(0, 50, 2000)
    aif = np.zeros_like(time)
    aif[(time > 5) & (time < 15)] = 1

    model.set_time(time)
    model.set_aif(aif)

    lamdalist = [4.2, 3.9, 0.1]

    for i, lam in enumerate(lamdalist):

        np.testing.assert_array_equal(model.convolution_w_exp(lam,
                                                              fftconvolution=True), model.convolution_w_exp(lam,
                                                                                                            fftconvolution=False), verbose=False)


def test_genericModel_readableParameters_contain_all_keys(preparedmodel):
    assert all([k in preparedmodel.readable_parameters for k in ("F", "v")])

def test_genericModel_fit_model_returns_bool(preparedmodel):
    """Test whether the fit routine reports sucess of fitting
    """

    return_value = preparedmodel.fit_model(np.asarray([1.,2.]))

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
    return_value = preparedmodel.fit_model(np.asarray([1.,2.]))

    assert np.allclose(preparedmodel._parameters, start_parameters)

