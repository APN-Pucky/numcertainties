import jacobi as jb
from smpl import io
import numpy as np

from numcertainties import unc
class nunc(unc):
# We keep a stack of operations until we need to evaluate the result
    def propagate(self):
        y,ycov=jb.propagate(self.stack, self.x, self.xcov)
        #print("ycov",ycov)
        return self.__class__(y,ycov)