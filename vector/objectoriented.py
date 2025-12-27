"""Vector class.

```python
>>> from vector import Vector
>>> Vector((1, 2, 3))
Vector(1, 2, 3, ...)
>>> Vector.randn(3)
Vector(-0.5613820142699765, -0.028308921297709365, 0.8270724508948077, ...)
>>> Vector(3)
Vector(0, 0, 0, 1, ...)
```

The immutable `Vector` class wraps all the mentioned functions into a tidy package, making them easier to use by enabling interaction through operators.

Its coefficients are internally stored as a tuple in the `coef` attribute and therefore *zero-indexed*.

Vector operations return the same type (`type(v+w)==type(v)`) so the class can easily be extended (to e.g. a polynomial class).
"""

from .functional import *



__all__ = ('Vector',)



class Vector:
    """An infinite-dimensional vector class.
    
    Its coefficients are internally stored as a tuple in the `coef` attribute.
    """
    
    ZERO: 'Vector'
    """Zero vector.
    
    See also
    --------
    [`veczero`][vector.functional.veczero]
    """
    
    #creation
    def __init__(self, coef):
        """Create a new vector with the given coefficients
        or the `i`-th basis vector if an integer `i` is given.
        
        Notes
        -----
        - varargs (single argument=basis or multiple args=coefficients)?
        No, because then a single coefficient vector can't be distinguished
        from an index for a basis vector.
        - Automatically trim on creation?
        Nah, other functions also don't do that.
        """
        if isinstance(coef, int):
            self.coef = vecbasis(coef)
        else:
            self.coef = tuple(coef)
    
    @staticmethod
    def rand(n):
        """Create a random vector of `n` uniform coefficients in `[0, 1[`.
        
        See also
        --------
        [`vecrand`][vector.functional.vecrand]
        """
        return Vector(vecrand(n))
    
    @staticmethod
    def randn(n, normed=True, mu=0, sigma=1):
        """Create a random vector of `n` normal distributed coefficients.
        
        See also
        --------
        [`vecrandn`][vector.functional.vecrandn]
        """
        return Vector(vecrandn(n, normed, mu, sigma))
    
    
    #sequence
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
        """Return if of same type with same coefficients.
        
        See also
        --------
        [`veceq`][vector.functional.veceq]
        """
        return isinstance(other, type(self)) and veceq(self, other)
    
    
    def __lshift__(self, other):
        """Return a vector with coefficients shifted to lower indices.
        
        See also
        --------
        [`veclshift`][vector.functional.veclshift]
        """
        return type(self)(self[other:])
    
    def __rshift__(self, other):
        """Return a vector with coefficients shifted to higher indices.
        
        See also
        --------
        [`vecrshift`][vector.functional.vecrshift]
        """
        return type(self)((0,)*other + self.coef)
    
    
    #utility
    def trim(self, tol=1e-9):
        """Remove all trailing near zero (abs<=tol) coefficients.
        
        See also
        --------
        [`vectrim`][vector.functional.vectrim]
        """
        return type(self)(vectrim(self, tol))
    
    def is_parallel(self, other):
        """Return if the other vector is parallel.
        
        See also
        --------
        [`vecparallel`][vector.functional.vecparallel]
        """
        return vecparallel(self, other)
    
    
    #Hilbert space
    def absq(self):
        """Return the sum of absolute squares of the coefficients.
        
        See also
        --------
        [`vecabsq`][vector.functional.vecabsq]
        """
        return vecabsq(self)
    
    def __abs__(self):
        """Return the Euclidean/L2-norm.
        
        Return the square root of `vecabsq`.
        
        See also
        --------
        [`vecabs`][vector.functional.vecabs]
        """
        return self.absq()**0.5
    
    def __matmul__(self, other):
        """Return the real dot product of two vectors.
        
        No argument is complex conjugated. All coefficients are used as is.
        
        See also
        --------
        [`vecdot`][vector.functional.vecdot]
        """
        return vecdot(self, other)
    
    
    #vector space operations like they would be correct on paper:
    #+v, -v, v+w, v-w, av, va, v/a, v//a
    def __pos__(self):
        """Return the unary positive.
        
        See also
        --------
        [`vecpos`][vector.functional.vecpos]
        """
        return type(self)(vecpos(self))
    
    def __neg__(self):
        """Return the negative.
        
        See also
        --------
        [`vecneg`][vector.functional.vecneg]
        """
        return type(self)(vecneg(self))
    
    def __add__(self, other):
        """Return the vector sum.
        
        See also
        --------
        [`vecadd`][vector.functional.vecadd]
        """
        return type(self)(vecadd(self, other))
    __radd__ = __add__
    
    def addc(self, c, i=0):
        """Return the sum with the `i`-th basis vector times `c`.
        
        See also
        --------
        [`vecaddc`][vector.functional.vecaddc]
        """
        return type(self)(vecaddc(self, c, i))
    
    def __sub__(self, other):
        """Return the vector difference.
        
        See also
        --------
        [`vecsub`][vector.functional.vecsub]
        """
        return type(self)(vecsub(self, other))
    def __rsub__(self, other):
        return type(self)(vecsub(other, self))
    
    def __mul__(self, other):
        """Return the scalar product.
        
        See also
        --------
        [`vecmul`][vector.functional.vecmul]
        """
        return type(self)(vecmul(other, self))
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        """Return the scalar true division.
        
        See also
        --------
        [`vectruediv`][vector.functional.vectruediv]
        """
        return type(self)(vectruediv(self, other))
    
    def __floordiv__(self, other):
        """Return the scalar floor division.
        
        See also
        --------
        [`vecfloordiv`][vector.functional.vecfloordiv]
        """
        return type(self)(vecfloordiv(self, other))
    
    def __mod__(self, other):
        """Return the elementwise mod with a scalar.
        
        See also
        --------
        [`vecmod`][vector.functional.vecmod]
        """
        return type(self)(vecmod(self, other))
    
    def __divmod__(self, other):
        """Return the elementwise divmod with a scalar.
        
        See also
        --------
        [`vecdivmod`][vector.functional.vecdivmod]
        """
        q, r = vecdivmod(self, other)
        return type(self)(q), type(self)(r)
    
    
    #elementwise
    def hadamard(self, other):
        """Return the elementwise product with another vector.
        
        See also
        --------
        [`vechadamard`][vector.functional.vechadamard]
        """
        return type(self)(vechadamard(self, other))
    
    def hadamardtruediv(self, other):
        """Return the elementwise true division with another vector.
        
        See also
        --------
        [`vechadamardtruediv`][vector.functional.vechadamardtruediv]
        """
        return type(self)(vechadamardtruediv(self, other))
    
    def hadamardfloordiv(self, other):
        """Return the elementwise floor division with another vector.
        
        See also
        --------
        [`vechadamardfloordiv`][vector.functional.vechadamardfloordiv]
        """
        return type(self)(vechadamardfloordiv(self, other))
    
    def hadamardmod(self, other):
        """Return the elementwise mod with another vector.
        
        See also
        --------
        [`vechadamardmod`][vector.functional.vechadamardmod]
        """
        return type(self)(vechadamardmod(self, other))
    
    def hadamardmin(self, other):
        """Return the elementwise minimum with another vector.
        
        See also
        --------
        [`vechadamardmin`][vector.functional.vechadamardmin]
        """
        return type(self)(vechadamardmin(self, other))
    
    def hadamardmax(self, other):
        """Return the elementwise maximum with another vector.
        
        See also
        --------
        [`vechadamardmax`][vector.functional.vechadamardmax]
        """
        return type(self)(vechadamardmax(self, other))
    
    
    #python stuff
    def __format__(self, format_spec):
        return 'Vector(' + ', '.join(format(c, format_spec) for c in self.coef) + ')'
    
    def __str__(self):
        return 'Vector(' + ', '.join(map(str, self.coef)) + ')'



Vector.ZERO = Vector(veczero)
