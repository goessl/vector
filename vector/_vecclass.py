from ._vecfunctions import *



__all__ = ['Vector']



class Vector:
    """An infinite-dimensional vector class.
    
    Its coefficients are internally stored as a tuple in the `coef` attribute.
    """
    __slots__ = ['coef']
    
    #construction stuff
    def __init__(self, coef):
        """Create a new vector with the given coefficients
        or the `i`-th basis vector if an integer `i` is given."""
        if isinstance(coef, int):
            self.coef = vecbasis(coef)
        else:
            self.coef = tuple(coef)
    
    @classmethod
    def rand(cls, n):
        """Create a random vector of `n` uniform coefficients in `[0, 1[`."""
        return cls(vecrand(n))
    
    @classmethod
    def randn(cls, n, normed=True, mu=0, sigma=1):
        """Create a random vector of `n` normal distributed coefficients."""
        return cls(vecrandn(n, normed=normed, mu=mu, sigma=sigma))
    
    
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
        return isinstance(other, type(self)) and veceq(self, other)
    
    
    def __lshift__(self, other):
        """Return a vector with coefficients shifted to lower indices."""
        return type(self)(self[other:])
    
    def __rshift__(self, other):
        """Return a vector with coefficients shifted to higher indices."""
        return type(self)((0,)*other + self.coef)
    
    
    #utility stuff
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
        return self.absq()**0.5
    
    def __matmul__(self, other):
        """Return the real dot product of two vectors.
        
        No argument is complex conjugated. All coefficients are used as is.
        """
        return vecdot(self, other)
    
    
    #vector space operations like they would be correct on paper:
    #v+w, v-w, av, va, v/a, v//a
    def __pos__(self):
        return type(self)(vecpos(self))
    
    def __neg__(self):
        return type(self)(vecneg(self))
    
    def __add__(self, other):
        """Return the vector sum."""
        return type(self)(vecadd(self, other))
    __radd__ = __add__
    
    def __sub__(self, other):
        """Return the vector difference."""
        return type(self)(vecsub(self, other))
    def __rsub__(self, other):
        return type(self)(vecsub(other, self))
    
    def __mul__(self, other):
        """Return the scalar product."""
        return type(self)(vecmul(other, self))
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        """Return the scalar true division."""
        return type(self)(vectruediv(self, other))
    
    def __floordiv__(self, other):
        """Return the scalar floor division."""
        return type(self)(vecfloordiv(self, other))
    
    def __mod__(self, other):
        """Return the elementwise mod with a scalar."""
        return type(self)(vecmod(self, other))
    
    
    #elementwise stuff
    def hadamard(self, other):
        """Return the elementwise product with another vector."""
        return type(self)(vechadamard(self, other))
    
    def hadamardtruediv(self, other):
        """Return the elementwise true division with another vector."""
        return type(self)(vechadamardtruediv(self, other))
    
    def hadamardfloordiv(self, other):
        """Return the elementwise floor division with another vector."""
        return type(self)(vechadamardfloordiv(self, other))
    
    def hadamardmod(self, other):
        """Return the elementwise mod with another vector."""
        return type(self)(vechadamardmod(self, other))
    
    def hadamardmin(self, other):
        """Return the elementwise minimum with another vector."""
        return type(self)(vechadamardmin(self, other))
    
    def hadamardmax(self, other):
        """Return the elementwise maximum with another vector."""
        return type(self)(vechadamardmax(self, other))
    
    
    #python stuff
    def __format__(self, format_spec):
        return 'Vector(' + ', '.join(format(c, format_spec) for c in self.coef) + ')'
    
    def __str__(self):
        return 'Vector(' + ', '.join(map(str, self.coef)) + ')'



Vector.ZERO = Vector(veczero)
"""Zero vector."""
