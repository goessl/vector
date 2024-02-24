from math import sqrt, isclose, hypot
from random import gauss
from itertools import starmap, zip_longest, repeat
from operator import add, sub, mul, truediv



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
        return v / abs(v) if normed else v
    
    
    
    #sequence stuff
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
        return isinstance(other, type(self)) and self.trim().coef == other.trim().coef
    
    
    def __lshift__(self, other):
        return type(self)(self[other:])
    
    def __rshift__(self, other):
        return type(self)(other*(0,) + self.coef)
    
    def trim(self, tol=1e-8):
        """Removes all trailing near zero (<=tol=1e-8) coefficients."""
        c = self.coef
        while c and abs(c[-1])<=tol:
            c = c[:-1]
        return type(self)(c)
    
    def round(self, ndigits=None):
        """Rounds all coefficients to the given precision."""
        return type(self)(round(c, ndigits) for c in self).trim()
    
    
    
    #Hilbert space stuff
    def __abs__(self):
        return hypot(*self)
    
    def __matmul__(self, other):
        #https://docs.python.org/3/library/itertools.html
        return sum(map(mul, self, other))
    
    
    
    #vector space operations
    #The return type of the arithmetic operations is determined by
    #the first argument (self) to enable correctly typed results for subclasses.
    #E.g. the sum of two HermiteFunctions should again be a HermiteFunction,
    #and not a Vector.
    @staticmethod
    def map_zip(f, v, w):
        """Applies f(v, w) elementwise; the second argument may be iterable."""
        try: #second argument iterable
            return type(v)(map(f, v, w))
        except TypeError: #second argument scalar
            return type(v)(map(f, v, repeat(w)))
    
    @staticmethod
    def map_zip_longest(f, v, w):
        """Applies f(v, w) elementwise; the second argument may be iterable."""
        try: #second argument iterable
            return type(v)(starmap(f, zip_longest(v, w, fillvalue=0)))
        except TypeError: #second argument scalar
            return type(v)(map(f, v, repeat(w)))
    
    #implementing vector space operations like they would be correct on paper:
    #v+w, v-w, av, va, v/a
    def __add__(self, other):
        return Vector.map_zip_longest(add, self, other)
    
    def __sub__(self, other):
        return Vector.map_zip_longest(sub, self, other)
    
    def __mul__(self, other):
        return Vector.map_zip(mul, self, other)
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        return Vector.map_zip(truediv, self, other)
    
    
    
    #python stuff
    def __str__(self):
        return 'Vector(' + ', '.join(map(str, self.coef)) + ', ...)'



Vector.ZERO = Vector(tuple())
