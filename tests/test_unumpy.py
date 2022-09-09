import pytest
import numpy as np
from numcertainties import semi_analytic_uncertainty as unc
from uncertainties import unumpy
unv = unumpy.nominal_values
std = unumpy.std_devs

def test_unumpy():
    x = [1, 2]
    xcov = [[0.03**2, 0],
	        [0, 0.04**2]]
    ua = unumpy.uarray(x,np.sqrt(np.diag(xcov)))
    uaua = ua **2
    n = unc(x, xcov)
    nn = (n**2).propagate()
    print (nn.get_value(), nn.get_std())
    assert np.allclose(nn.get_value() , unv(uaua))
    print(nn.get_std(),std(uaua) )
    assert np.allclose(nn.get_std() , std(uaua))