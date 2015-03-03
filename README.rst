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

Using this library is simple: ::

  import compartmentmodels.compartmentmodels as CM
  t,c,a=CM.loaddata('tests/lung.csv')
  mm=CM.CompartmentModel(time=t, curve=c, aif=a)
  mm.fit_model(startdict={'F':100., 'v':10.})
  print mm.get_parameters()

A demonstration can also be found in the accompanying ipython notebook.

Development
---------

* fork the repository

* Look for an issue that needs to be adressed

* write a test for this issue in the tests/ directory (hint: simply extend one of the test_* files) 

* Run the tests: py.test
  
* modify the code until the test passes

* commit and file a pull request 
  
Note: Currently, the developers have not much experience with this form of collaboration. Please contact us if you wish to contribute, we are more than happy about contributors, and we are willing to learn :)

Features
--------
At the time being, this package provides  implementations of several tracer-kinetic models. Currently, this module is being developed, following a test-driven approach (well, we try).

Next milestone
---------

The next milestone will be Release 0.1, with working implementations of the two-compartment exchange model, the compartment uptake model and the one-compartment model.


