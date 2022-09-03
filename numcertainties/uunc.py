from numcertainties import unc
import numpy as np
import uncertainties
from uncertainties import unumpy
import itertools

class uunc(unc):
# We keep a stack of operations until we need to evaluate the result
    def propagate(self):
	# TODO maybe more complicted than this for higher dimensions
        ux = uncertainties.correlated_values(self.x,self.xcov)
        y = self.stack(np.array([*ux]))
        #y = self.stack(unumpy.uarray(unumpy.nominal_values(np.array([*ux])),unumpy.std_devs(np.array([*ux]))))
        #print("ux" ,ux)
        #print("y" ,y,y.__class__)
        ycov = uncertainties.covariance_matrix([*y])
        #print ("ycov", ycov)
        return self.__class__(y,ycov)