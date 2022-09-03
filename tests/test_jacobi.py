import jacobi as jb
import pytest 
import numpy as np
from numcertainties import nunc

def test_jacobi():
	x = [1, 2]
	xcov = [[3, 1],
	        [1, 4]]

	y, ycov = jb.propagate(lambda x:x**2, x, xcov)
	z,zcov = jb.propagate(lambda x:x**2, y, ycov)
	a,acov = jb.propagate(lambda x:x**4, x, xcov)
	print (a , z)
	print(acov , zcov)
#	assert np.all(a == z)
#	assert np.all(acov == zcov)

	n = nunc(x, xcov)
	nn = n**4
#	print("n",n)
	print("nn",nn)
test_jacobi()