from math import sqrt, isclose, hypot, sumprod
from random import random, gauss
from itertools import starmap, zip_longest, repeat
from operator import add, sub, mul, truediv, floordiv, eq



#creation stuff
veczero = ()
"""Zero vector."""

def vecbasis(i, c=1):
    """Return the `i`-th basis vector times `c`.
    
    The retured value is a tuple with `i` integer zeros followed by `c`.
    """
    return (0,)*i + (c,)

def vecrandom(n):
    """Return a random vector of `n` uniform coefficients in `[0, 1[`."""
    return tuple(random() for _ in range(n))

def vecgauss(n, normed=True, mu=0, sigma=1):
    """Return a random vector of `n` normal distributed coefficients."""
    v = tuple(gauss(mu=mu, sigma=sigma) for _ in range(n))
    return vectruediv(v, vecabs(v)) if normed else v


#sequence stuff
def veceq(v, w):
    """Return if two vectors are equal."""
    return all(starmap(eq, zip_longest(v, w, fillvalue=0)))

def vectrim(v, tol=1e-9):
    """Remove all trailing near zero (abs(v_i)<=tol) coefficients."""
    while v and abs(v[-1])<=tol:
        v = v[:-1]
    return v

def vecround(v, ndigits=None):
    """Round all coefficients to the given precision."""
    return tuple(round(c, ndigits) for c in v)


#Hilbert space stuff
def vecabsq(v):
    """Return the sum of absolute squares of the coefficients."""
    #return sumprod(v, v)
    #avoid abs twice and exponentiation
    #return sum((avi:=abs(vi))*avi for vi in v)
    #walruss doesn't save memory, as generators are lazy
    return sum(avi*avi for avi in (abs(vi) for vi in v))

def vecabs(v):
    """Return the Euclidean/L2-norm.
    
    Return the square root of `vecabsq`.
    """
    #hypot(*v) doesn't work for complex
    #math.sqrt doesn't work for complex and cmath.sqrt always returns complex
    #therefore use **0.5 instead of sqrt because it is type conserving
    return vecabsq(v)**0.5

def vecdot(v, w):
    """Return the real dot product of two vectors.
    
    No argument is complex conjugated. All coefficients are used as is.
    """
    #unreadable:
    #return sumprod(v[:min(len(v), len(w))], w[:min(len(v), len(w))])
    #return sumprod(*zip(*zip(v, w))) would be more precise, but is bloat
    return sum(map(mul, v, w))


#vector space stuff
def vecadd(*vs):
    """Return the sum of vectors."""
    return tuple(map(sum, zip_longest(*vs, fillvalue=0)))

def vecsub(v, w):
    """Return the difference of two vectors."""
    return tuple(starmap(sub, zip_longest(v, w, fillvalue=0)))

def vecmul(a, v):
    """Return the product of a scalar and a vector."""
    return tuple(map(mul, repeat(a), v))

def vectruediv(v, a):
    """Return the true division of a vector and a scalar."""
    return tuple(map(truediv, v, repeat(a)))

def vecfloordiv(v, a):
    """Return the floor division of a vector and a scalar."""
    return tuple(map(floordiv, v, repeat(a)))



class Vector:
    """An infinite-dimensional vector class.
    
    Its coefficients are internally stored as a tuple in the `coef` attribute.
    """
    
    #construction stuff
    def __init__(self, coef):
        """Create a new vector with the given coefficients
        or the `i`-th basis vector if an integer `i` is given."""
        if isinstance(coef, int):
            self.coef = vecbasis(coef)
        else:
            self.coef = tuple(coef)
    
    @staticmethod
    def random(n):
        """Create a random vector of `n` uniform coefficients in `[0, 1[`."""
        return Vector(vecrandom(n))
    
    @staticmethod
    def gauss(n, normed=True, mu=0, sigma=1):
        """Create a random vector of `n` normal distributed coefficients."""
        return Vector(vecgauss(n, normed=normed, mu=mu, sigma=sigma))
    
    
    #sequence stuff
    def __len__(self):
        """Return the number of set coefficients."""
        return len(self.coef)
    
    def __getitem__(self, key):
        """Return the indexed coefficient or coefficients.
        
        Not set coefficients default to 0.
        """
        #getter has if and try-else
        #might access coef directly in the following methods
        #for better performance
        #on the other hand most functions access the iterator
        #that directly goes for the tuple
        if isinstance(key, slice):
            #enable stop>=len with zero padding
            if key.stop is not None and key.stop >= 0:
                l = key.stop
            else:
                l = len(self)
            return Vector(self[i] for i in range(*key.indices(l)))
        try:
            return self.coef[key]
        except IndexError:
            return 0
    
    def __iter__(self):
        """Return an iterator over the set coefficients."""
        return iter(self.coef)
    
    def __eq__(self, other):
        """Return if of same type with same coefficients."""
        #maybe check isinstance(other, Vector)?
        return isinstance(other, type(self)) and veceq(self.coef, other.coef)
    
    
    def __lshift__(self, other):
        """Return a vector with coefficients shifted to lower indices."""
        return type(self)(self[other:])
    
    def __rshift__(self, other):
        """Return a vector with coefficients shifted to higher indices."""
        return type(self)(other*(0,) + self.coef)
    
    def trim(self, tol=1e-9):
        """Remove all trailing near zero (abs<=tol) coefficients."""
        return type(self)(vectrim(self))
    
    def round(self, ndigits=None):
        """Round all coefficients to the given precision."""
        return type(self)(vecround(self))
    
    
    #Hilbert space stuff
    def absq(self):
        """Return the sum of absolute squares of the coefficients."""
        return vecabsq(self)
    
    def __abs__(self):
        """Return the Euclidean/L2-norm.
        
        Return the square root of `vecabsq`.
        """
        return vecabs(self)
    
    def __matmul__(self, other):
        """Return the real dot product of two vectors.
        
        No argument is complex conjugated. All coefficients are used as is.
        """
        return vecdot(self, other)
    
    
    #vector space operations like they would be correct on paper:
    #v+w, v-w, av, va, v/a, v//a
    def __add__(self, other):
        """Return the vector sum."""
        return Vector(vecadd(self, other))
    
    def __sub__(self, other):
        """Return the vector difference."""
        return Vector(vecsub(self, other))
    
    def __mul__(self, other):
        """Return the scalar product."""
        return Vector(vecmul(other, self))
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        """Return the scalar true division."""
        return Vector(vectruediv(self, other))
    
    def __floordiv__(self, other):
        """Return the scalar floor division."""
        return Vector(vecfloordiv(self, other))
    
    
    #python stuff
    def __str__(self):
        return 'Vector(' + ', '.join(map(str, self.coef)) + ', ...)'



Vector.ZERO = Vector(veczero)
"""Zero vector."""
