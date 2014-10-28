import pytest

# test wether pytest runs at all
def func(x):
    return x+1

def test_func():
    assert func(3)== 5

# set up a fixture for the Generic model
@pytest.fixture
def model():
    from compartmentmodels.GenericModel import GenericModel
    model=GenericModel()
    return model

def test_genericModel_has_string_representation(model):
    assert model.__str__() == "Generic model"

