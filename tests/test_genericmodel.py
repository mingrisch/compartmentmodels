import pytest
import scipy as sp
import numpy as np


# test wether pytest runs at all
def func(x):
    return x + 1


def test_func():
    assert func(3) == 5

# set up a fixture for the Generic model


@pytest.fixture(scope='module')
def model():
    from compartmentmodels.GenericModel import GenericModel
    model = GenericModel()
    return model


def test_genericModel_has_string_representation(model):
    assert model.__str__() == "Generic model"


def test_genericModel_set_and_get_time(model):
    time = sp.linspace(0, 20, 50)
    model.set_time(time)
    np.testing.assert_array_equal(model.get_time(), time)
    
def test_genericModel_set_and_get_aif(model):
    aif = np.random.randn(50)
    model.set_aif(aif)
    np.testing.assert_array_equal(model.get_aif(), aif)
    
def test_genericModel_set_and_get_curve(model):
    curve = np.random.rand(50)
    model.set_curve(curve)
    np.testing.assert_array_equal(model.get_curve(), curve)
    
def test_genericModel_readableParameters_contain_all_keys(model):
    assert all(k in model.readable_parameters for k in  ("F", "v", "MTT"))

