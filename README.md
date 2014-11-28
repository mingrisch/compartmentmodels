===============================
compartmentmodels
===============================

.. image:: https://badge.fury.io/py/compartmentmodels.png
    :target: http://badge.fury.io/py/compartmentmodels

.. image:: https://travis-ci.org/michimichi/compartmentmodels.png?branch=develop
        :target: https://travis-ci.org/michimichi/compartmentmodels

.. image:: https://pypip.in/d/compartmentmodels/badge.png
        :target: https://pypi.python.org/pypi/compartmentmodels


Some kinetic models for DCE MRI

* Free software: BSD license

Installation 
----------
A virtual environment is useful for installing this package.
Currently, we need numpy, scipy and pytest. Quite possibly, more dependencies will arise.

Installation:

* create a virtual environment, following your preferred approach
* clone this repository
* install the required dependencies: pip install -r requirements.txt
* After checking out and installing the required dependencies, it may be helpful (and possibly required) to install the package locally by: pip install -e .

Usage
------
```
import compartmentmodels.compartmentmodels as CM
t,c,a=CM.loaddata('tests/lung.csv')
mm=CM.CompartmentModel(time=t, curve=c, aif=a)
mm.fit_model(startdict={'F':100., 'v':10.})
print mm.get_parameters()


```
Development
---------

* Look for an issue that needs to be adressed

* write a test for this issue in the tests/ directory (hint: simply extend tests/test_genericmodel.py)

* Run the tests: py.test
  
* modify the code until the test passes

* and commit 

Features
--------
At the time being, this package provides a simple implementation of a one-compartment model. Currently, this module is under heavy development which follows, at least in parts, a test-driven approach.

Next milestone
---------
The next (and first) milestone is to reach a working implementation that does not depend on external numeric packages.
