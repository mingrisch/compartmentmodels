# test wether pytest runs at all
def func(x):
    return x+1

def test_func():
    assert func(3)== 5

