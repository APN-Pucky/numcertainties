import jacobi as jb
from smpl import io
import numpy as np

class nunc:
#We keep a stack of operations until we need to evaluate the result
    def __init__(self, x, xcov,stack=None):
        self.x= np.array(x)
        self.xcov= np.atleast_1d(xcov)
        if stack is None:
            self.stack= lambda x:x
        else:
            self.stack= stack

    def propagate(self):
        y,ycov=jb.propagate(self.stack, self.x, self.xcov)
        return self.__class__(y,ycov)

    def get_value(self):
        return self.propagate().x

    def get_cov(self):
        return self.propagate().xcov

    def __pow__(self, other, modulo=None):
        return self.__class__(self.x,self.xcov, lambda x:self.stack(x)**other)

    def __rpow__(self, other, modulo=None):
        return self.__class__(self.x,self.xcov, lambda x:other**self.stack(x))

    def __radd__(self, other):
        return self.__class__(self.x,self.xcov, lambda x:other+self.stack(x))

    def __add__(self, other):
        return self.__class__(self.x,self.xcov, lambda x:self.stack(x)+other)

    def __rsub__(self, other):
        return self.__class__(self.x,self.xcov, lambda x:other-self.stack(x))

    def __sub__(self, other):
        return self.__class__(self.x,self.xcov, lambda x:self.stack(x)-other)

    def __rmul__(self, other):
        return self.__class__(self.x,self.xcov, lambda x:other*self.stack(x))

    def __mul__(self, other):
        return self.__class__(self.x,self.xcov, lambda x:self.stack(x)*other)

    def __rtruediv__(self, other):
        return self.__class__(self.x,self.xcov, lambda x:other/self.stack(x))

    def __truediv__(self, other):
        return self.__class__(self.x,self.xcov, lambda x:self.stack(x)/other)

    def __str__(self):
        return str(self.get_value()) + "[" + str(self.get_cov()) + "]"

    def __repr__(self):
        return str(self.get_value()) + "[" + str(self.get_cov()) + "]"

    def __format__(self, fmt):
        return str(self.get_value()) + "[" + str(self.get_cov()) + "]"