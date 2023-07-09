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
            self.coef = (0,)*coef + (1,)
        else:
            self.coef = tuple(coef)
    
    @staticmethod
    def random(deg, normed=True):
        """Creates a Hermite function series of the given degree
        with normal distributed coefficients."""
        coef = np.random.normal(size=deg+1)
        if normed:
            coef /= np.linalg.norm(coef)
        return HermiteFunction(coef)
    
    @staticmethod
    def fit(x, y, deg):
        """Creates a least squares Hermite function series fit
        with the given degree for the given x and y values."""
        #https://de.wikipedia.org/wiki/Multiple_lineare_Regression
        return HermiteFunction(c * np.sqrt(2**i*factorial(i)*np.sqrt(np.pi))
                for i, c in enumerate(hermfit(x, y/np.exp(-x**2/2), deg)))
    
    
    
    #container stuff
    def __len__(self):
        return len(self.coef)
    
    def __getitem__(self, key):
        try:
            return self.coef[key]
        except IndexError:
            return 0
    
    def __iter__(self):
        return iter(self.coef)
    
    def __eq__(self, other):
        return self.coef == other.coef
    
    
    def __lshift__(self, other):
        return HermiteFunction(self[1:])
    
    def __rshift__(self, other):
        return HermiteFunction((0,) + self.coef)
    
    
    
    #Hilbert space stuff
    def __abs__(self):
        return np.sqrt(self @ self)
    
    def __matmul__(self, other):
        #https://docs.python.org/3/library/itertools.html
        return sum(starmap(operator.mul, zip(self, other)))
    
    
    
    #vector space operations
    @staticmethod
    def map_zip(f, v, w):
        """Applies f(v, w) elementwise if possible,
                otherwise elementwise in the first argument."""
        try: #second argument iterable
            return HermiteFunction(f(a, b) for a, b in zip(v, w))
        except TypeError: #second argument scalar
            return HermiteFunction(f(c, w) for c in v)
    
    @staticmethod
    def map_zip_longest(f, v, w):
        """Applies f(v, w) elementwise if possible,
                otherwise elementwise in the first argument."""
        try: #second argument iterable
            return HermiteFunction(f(a, b)
                    for a, b in zip_longest(v, w, fillvalue=0))
        except TypeError: #second argument scalar
            return HermiteFunction(f(c, w) for c in v)
    
    #implement vector space operations like they would be correct on paper:
    #v+w, v-w, av, va, v/a
    def __add__(self, other):
        return HermiteFunction.map_zip_longest(operator.add, self, other)
    
    def __sub__(self, other):
        return HermiteFunction.map_zip_longest(operator.sub, self, other)
    
    def __mul__(self, other):
        return HermiteFunction.map_zip(operator.mul, self, other)
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        return HermiteFunction.map_zip(operator.truediv, self, other)
    
    
    
    #function stuff
    @property
    def deg(self):
        """Degree of this series (index of the highest set coefficient)."""
        return len(self) - 1
    
    def __call__(self, x):
        return np.exp(-x**2/2) \
                * sum(c / np.sqrt(2**i * factorial(i) * np.sqrt(np.pi))
                * hermval(x, (0,)*i+(1,))
                for i, c in enumerate(self))
    
    def der(self, n=1):
        """Returns the n-th derivative of this series."""
        res = self
        for _ in range(n):
            res = (np.sqrt((i+1)/2) for i in range(len(res)-1)) * (res<<1) \
                    - (np.sqrt(i/2) for i in range(len(res)+1)) * (res>>1)
        return res
    
    @cached_property
    def kin(self):
        """The kinetic energy of this series."""
        #return -1/2 * self.dot(self.der(2))
        return abs(self.der())**2 / 2
    
    
    
    #python stuff
    def __str__(self):
        s = f'{self[0]:.1f} h_0'
        for i, c in enumerate(self[1:]):
            s += f' + {c:.1f} h_{i+1}'
        return s
