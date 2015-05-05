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
def TwoCXM():
    from compartmentmodels.compartmentmodels import loaddata, savedata
    from compartmentmodels.compartmentmodels import TwoCXModel
    # need to set a stardict and convert input parameter maybe.
    # Set parameters for the model
    startdict = {'Fp': 31.0, 'vp': 11.2, 'PS': 4.9, 've': 13.2}
    time, aif1, aif2 = loaddata(filename='tests/cerebralartery.csv')
    aif = aif1 - aif1[0:5].mean()
    model = TwoCXModel(time=time, curve=aif, aif=aif, startdict=startdict)
    # calculate a model curve
    model.curve = model.calc_modelfunction(model._parameters)
    model.curve += 0.002 * model.curve.max() * np.random.randn(len(time))
    # number of bootstraps
    model.k = 100

    return model


def test_2CXM_fitting(TwoCXM):
    """ Does the fit return reasonabe estimates?
    """
    assert len(TwoCXM.startdict) == 4
    true_values = TwoCXM.startdict.copy()
    assert true_values == {'Fp': 31.0, 'vp': 11.2, 'PS': 4.9, 've': 13.2}
    fit_result = TwoCXM.fit_model()
    fitparameters = TwoCXM.get_parameters()
    # assert number of parameters
    assert len(TwoCXM._parameters) == 4
    assert len(TwoCXM.readable_parameters) == 8

    # assert number of fit parameters (t4)

    # to do : compare the fit results (<10% )
    for parameter in ['Fp', 'vp', 'PS', 've']:

        fit = fitparameters.get(parameter)
        true = true_values.get(parameter)

        assert (fit - true) / true < 0.1


# def test_exchangemodel_output(TwoCXM):
#     """Is 'low estimate' < 'mean estimate' < 'high estimate'?
#     Are fittet Parameters in between 'low estimate' and 'high estimate'?
#     """
#     fit_result= TwoCXM.fit_model()
#     bootstrap = TwoCXM.bootstrap()
#     assert (TwoCXM._bootstrapped == True)
#     dict_fit={'Fp':TwoCXM.readable_parameters['Fp'],
#                 'vp':TwoCXM.readable_parameters['vp'],
#                 'TP':TwoCXM.readable_parameters['TP'],
#                 'E':TwoCXM.readable_parameters['E'],
#                 'PS':TwoCXM.readable_parameters['PS'],
#                 've':TwoCXM.readable_parameters['ve'],
#                 'TE':TwoCXM.readable_parameters['TE'],
#                     }
#     assert (TwoCXM.readable_parameters['low estimate'] <
#             TwoCXM.readable_parameters['mean estimate'])
#     assert (TwoCXM.readable_parameters['mean estimate'] <
#             TwoCXM.readable_parameters['high estimate'])
#     assert (TwoCXM.readable_parameters['low estimate'] < dict_fit)
#     assert (dict_fit < TwoCXM.readable_parameters['high estimate'])
