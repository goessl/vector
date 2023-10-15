from math import sqrt, isclose
from random import gauss
from itertools import starmap, zip_longest
import operator



class Vector:
    """An infinite dimensional vector class."""
    
    #construction stuff
    def __init__(self, coef):
        """Creates a new vector with the given coefficients
        or the i-th basis vector if an integer i is given."""
        if isinstance(coef, int):
            self.coef = Vector.basis_tuple(coef)
        else:
            self.coef = tuple(coef)
    
    @staticmethod
    def basis_tuple(i):
        return (0,)*i + (1,)
    
    @staticmethod
    def random(n, normed=True):
        """Creates a, by default normed, vector of the given dimensionality
        with normal distributed coefficients."""
        v = Vector(gauss() for _ in range(n))
        if normed:
            v /= abs(v)
        return v
    
    
    
    #container stuff
    def __len__(self):
        return len(self.coef)
    
    def __getitem__(self, key):
        #TODO: slicing
        try:
            return self.coef[key]
        except IndexError:
            return 0
    
    def __iter__(self):
        return iter(self.coef)
    
    def __eq__(self, other):
        return self.coef == other.coef
    
    
    def __lshift__(self, other):
        return type(self)(self[other:])
    
    def __rshift__(self, other):
        return type(self)(other*(0,) + self.coef)
    
    def trim(self):
        """Removes all trailing near zero coefficients."""
        c = self.coef
        while c and isclose(c[-1], 0):
            c = c[:-1]
        return Vector(c)
    
    
    
    #Hilbert space stuff
    def __abs__(self):
        return sqrt(self @ self)
    
    def __matmul__(self, other):
        #https://docs.python.org/3/library/itertools.html
        return sum(starmap(operator.mul, zip(self, other)))
    
    
    
    #vector space operations
    #The return type of the arithmetic operations is determined by
    #the first argument (self) to enable correctly typed results for subclasses.
    #E.g. the sum of two HermiteSeries should again be a HermiteSeries,
    #and not a Vector.
    @staticmethod
    def map_zip(f, v, w):
        """Applies f(v, w) elementwise if possible,
        otherwise elementwise in the first argument."""
        try: #second argument iterable
            return type(v)(f(a, b) for a, b in zip(v, w))
        except TypeError: #second argument scalar
            return type(v)(f(c, w) for c in v)
    
    @staticmethod
    def map_zip_longest(f, v, w):
        """Applies f(v, w) elementwise if possible,
        otherwise elementwise in the first argument."""
        try: #second argument iterable
            return type(v)(f(a, b)
                    for a, b in zip_longest(v, w, fillvalue=0))
        except TypeError: #second argument scalar
            return type(v)(f(c, w) for c in v)
    
    #implement vector space operations like they would be correct on paper:
    #v+w, v-w, av, va, v/a
    def __add__(self, other):
        return Vector.map_zip_longest(operator.add, self, other)
    
    def __sub__(self, other):
        return Vector.map_zip_longest(operator.sub, self, other)
    
    def __mul__(self, other):
        return Vector.map_zip(operator.mul, self, other)
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        return Vector.map_zip(operator.truediv, self, other)
    
    
    
    #python stuff
    def __str__(self):
        return 'Vector(' + ', '.join(map(str, self.coef)) + ', 0, ...)'



Vector.ZERO = Vector(tuple())
