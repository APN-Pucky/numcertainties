import collections
import numpy as np
import uncertainties
from uncertainties import unumpy
import itertools

from numcertainties.base import base_uncertainty

class semi_analytic_uncertainty(base_uncertainty):
# We keep a stack of operations until we need to evaluate the result
	def _propagate(self):
		# TODO maybe more complicted than this for higher dimensions
		if len(self.x)>1:
			ux = uncertainties.correlated_values(self.x,self.xcov)
		else:
			ux = [uncertainties.ufloat(self.x,np.sqrt(self.xcov))]
		y = self.stack(np.array([*ux]))
 	       #y = self.stack(unumpy.uarray(unumpy.nominal_values(np.array([*ux])),unumpy.std_devs(np.array([*ux]))))
 	       #print("ux" ,ux)
 	       #print("y" ,y,y.__class__)
		ycov = uncertainties.covariance_matrix([*y])
 	       #print ("ycov", ycov)
		return self.__class__(unumpy.nominal_values(y),ycov,**self.params)