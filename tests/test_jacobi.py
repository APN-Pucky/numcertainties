import jacobi as jb
import pytest 
import numpy as np
from numcertainties import nunc
from numcertainties import uunc

def test_jacobi():
	x = [1, 2]
	xcov = [[3, 1],
	        [1, 4]]

	y, ycov = jb.propagate(lambda x:x**2, x, xcov)
	z,zcov = jb.propagate(lambda x:x**2, y, ycov)
	a,acov = jb.propagate(lambda x:x**4, x, xcov)
	assert np.allclose(a , z)
	assert np.allclose(acov , zcov) 

	n = nunc(x, xcov)
	nn = n**2

	assert np.allclose(nn.get_value() , y)
	assert np.allclose(nn.get_cov() , ycov)

	nn = nn**2

	assert np.allclose(nn.get_value() , z)
	assert np.allclose(nn.get_cov() , zcov)

	n = uunc(x, xcov)
	nn = n**4
	assert np.allclose(nn.get_value() , a)
	assert np.allclose(nn.get_cov() , acov)

test_jacobi()