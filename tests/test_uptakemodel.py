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
def TwoCUM():
    from compartmentmodels.compartmentmodels import loaddata, savedata
    from compartmentmodels.compartmentmodels import TwoCUModel
    # need to set a stardict and convert input parameter maybe.
    # Set parameters for the model
    startdict = {'Fp': 31.0, 'vp': 11.2, 'PS': 4.9}
    time, aif1, aif2 = loaddata(filename='tests/cerebralartery.csv')
    aif = aif1 - aif1[0:5].mean()
    model = TwoCUModel(time=time, curve=aif, aif=aif, startdict=startdict)
    # calculate a model curve
    model.curve = model.calc_modelfunction(model._fitparameters)
    model.curve += 0.002 * model.curve.max() * np.random.randn(len(time))
    # number of bootstraps
    model.k = 100

    return model


def test_2CUM_fitting(TwoCUM):
    """ Does the fit return reasonabe estimates?
    """
    assert len(TwoCUM.startdict) == 3
    true_values = TwoCUM.startdict.copy()
    assert true_values == {'Fp': 31.0, 'vp': 11.2, 'PS': 4.9}
    fit_result = TwoCUM.fit_model()
    fitparameters = TwoCUM.get_parameters()
    # assert number of parameters
    assert len(TwoCUM._fitparameters) == 3
    assert len(TwoCUM.phys_parameters) == 6

    # assert number of fit parameters (t4)

    # to do : compare the fit results (<10% )
    for parameter in ['Fp', 'vp', 'PS']:

        fit = fitparameters.get(parameter)
        true = true_values.get(parameter)

        assert (fit - true) / true < 0.1
