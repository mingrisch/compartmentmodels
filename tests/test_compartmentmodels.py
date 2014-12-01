#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_compartmentmodels
----------------------------------

Tests for `compartmentmodels` module.

The tests for the individual models are in separate files.

"""
import pytest
#import tempfile
import os
import numpy as np



from compartmentmodels.compartmentmodels import loaddata, savedata

def test_load_and_save(tmpdir):
    time = np.linspace(0,100)
    curve= np.random.randn(len(time))
    aif = np.random.randn(len(time))

    filename = os.path.join(str(tmpdir), 'tempfile.tca')
    
   # filename = tempfile.NamedTemporaryFile() 
    print filename

    savedata(filename, time, curve, aif)
    t, c, a = loaddata(filename)

    assert np.all(np.equal(time, t))
    assert np.all(np.equal(curve, c))
    assert np.all(np.equal(aif, a))

    

    
    
    
    
    
    
    