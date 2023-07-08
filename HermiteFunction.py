import numpy as np
from numpy.polynomial.hermite import hermval, hermfit
from scipy.special import factorial, binom
from itertools import starmap, zip_longest
import operator
from functools import cached_property



class HermiteFunction:
    """A Hermite function series class."""
    
    #construction stuff
    def __init__(self, coef):
        """Creates a new Hermite function series with the given coefficients
        or the i-th basis vector if an index is given."""
        if isinstance(coef, int):
            coef = [0]*coef + [1]
        self.coef = np.array(coef)
    
    @staticmethod
    def random(deg, normed=True):
        """Creates a Hermite function series of the given degree
        with random coefficients in [-1, 1[."""
        coef = np.random.uniform(-1, +1, deg+1)
        if normed:
            coef /= np.linalg.norm(coef)
        return HermiteFunction(coef)
    
    @staticmethod
    def fit(x, y, deg):
        """Creates a least squares Hermite function series fit
        with the given degree for the given x and y values."""
        #https://de.wikipedia.org/wiki/Multiple_lineare_Regression
        return HermiteFunction(tuple(c * np.sqrt(2**i*factorial(i)*np.sqrt(np.pi))
                for i, c in enumerate(hermfit(x, y/np.exp(-x**2/2), deg))))
    
    
    #Hilbert space stuff
    def dot(self, other):
        """Returns the L_R^2 dot product of self with other."""
        return np.vdot(self.coef[:len(other.coef)], \
                      other.coef[:len(self.coef)])
    
    def __abs__(self):
        return np.linalg.norm(self.coef)
    
    def __mul__(self, other):
        if isinstance(other, HermiteFunction):
            return HermiteFunction(
                    self.coef[:len(other.coef)] * other.coef[:len(self.coef)])
        else:
            return HermiteFunction(self.coef * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        if isinstance(other, HermiteFunction):
            return HermiteFunction([a+b for a, b \
                in zip_longest(self.coef, other.coef, fillvalue=0)])
        else:
            return HermiteFunction(self.coef + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    
    #function stuff
    @property
    def deg(self):
        """Degree of this series (index of the highest set coefficient)."""
        return len(self) - 1
    
    def __call__(self, x):
        return np.exp(-x**2/2) \
                * sum(c / np.sqrt(2**i * factorial(i) * np.sqrt(np.pi))
                * hermval(x, [0]*i+[1])
                for i, c in enumerate(self.coef))
    
    def der(self, n=1):
        """Returns the n-th derivative of this series."""
        coef = self.coef
        for _ in range(n):
            i = np.arange(len(coef)+1)
            coef = np.append(coef[1:], [0, 0])*np.sqrt((i+1)/2) \
                    - np.append([0], coef)*np.sqrt(i/2)
        return HermiteFunction(coef)
    
    def prod_reorder(self, other):
        """Returns the product of self and other, divided by h_0."""
        coef = np.zeros(len(self.coef)+len(other.coef)-1)
        for b in range(len(coef)):
            for n in range(b, len(self.coef)+len(other.coef)-1, 2):
                for d in range(-b, b+1, 2):
                    i, j = (n-d)//2, (n+d)//2
                    k = (n-b)//2
                    if 0<=i<len(self.coef) and 0<=j<len(other.coef):
                        coef[b] += self.coef[i] * other.coef[j] \
                            * factorial(k) * binom(i, k) * binom(j, k) \
                            * np.sqrt(factorial(b)/(factorial(i)*factorial(j)))
        return HermiteFunction(coef)
    
    @cached_property
    def kin(self):
        """The kinetic energy of this series."""
        #return -1/2 * self.dot(self.der(2))
        return abs(self.der())**2 / 2
    
    
    
    #python stuff
    def __str__(self):
        s = f'{self.coef[0]:.1f} h_0'
        for i, c in enumerate(self.coef[1:]):
            s += f' + {c:.1f} h_{i+1}'
        return s
