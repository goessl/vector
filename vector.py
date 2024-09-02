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

def vecrand(n):
    """Return a random vector of `n` uniform coefficients in `[0, 1[`."""
    return tuple(random() for _ in range(n))

def vecrandn(n, normed=True, mu=0, sigma=1):
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
    """Return the inner product of two vectors without conjugation."""
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



if __name__ == '__main__':
    from math import isclose, sqrt
    from random import random
    
    
    #vecbasis
    assert vecbasis(2, c=5) == (0, 0, 5)
    
    #vecrand
    vecrand(2)
    
    #vecrandn
    assert isclose(vecabsq(vecrandn(10)), 1)
    assert not isclose(vecabsq(vecrandn(10, normed=False)), 1)
    
    
    #veceq
    assert veceq((), ())
    assert veceq((0,), ())
    assert veceq((0, 0), (0,))
    assert veceq((1,), (1, 0))
    assert not veceq((1,), ())
    
    #vectrim
    assert vectrim(()) == ()
    assert vectrim((0,)) == ()
    assert vectrim((1, 0)) == (1,)
    
    #vecround
    assert vecround(()) == ()
    assert vecround((1.1,)) == (1,)
    
    
    #vecabssq
    assert vecabsq(()) == 0
    assert vecabsq((1j, 2, 3j)) == 14
    
    #vecabs
    assert vecabs(()) == 0
    assert vecabs((1,)) == 1
    assert vecabs((3, 4)) == 5
    
    #vecdot
    assert vecdot((), ()) == 0
    assert vecdot((1,), ()) == 0
    assert vecdot((1, 2), (3, 4)) == 11
    
    
    #vecadd
    assert vecadd() == ()
    assert vecadd((1, 2)) == (1, 2)
    assert vecadd((1, 2, 3), (4, 5)) == (5, 7, 3)
    
    #vecsub
    assert vecsub((), ()) == ()
    assert vecsub((1,), ()) == (1,)
    assert vecsub((), (1,)) == (-1,)
    assert vecsub((1, 2, 3), (4, 5)) == (-3, -3, 3)
    
    #vectruediv
    assert vectruediv((), 1) == ()
    assert vectruediv((4,), 2) == (2,)
    
    #vecfloordiv
    assert vecfloordiv((), 1) == ()
    assert vecfloordiv((3,), 2) == (1,)





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
    def rand(n):
        """Create a random vector of `n` uniform coefficients in `[0, 1[`."""
        return Vector(vecrand(n))
    
    @staticmethod
    def randn(n, normed=True, mu=0, sigma=1):
        """Create a random vector of `n` normal distributed coefficients."""
        return Vector(vecrandn(n, normed=normed, mu=mu, sigma=sigma))
    
    
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
        return type(self)(vecadd(self, other))
    
    def __sub__(self, other):
        """Return the vector difference."""
        return type(self)(vecsub(self, other))
    
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
    
    
    #python stuff
    def __str__(self):
        return 'Vector(' + ', '.join(map(str, self.coef)) + ', ...)'



Vector.ZERO = Vector(veczero)
"""Zero vector."""



if __name__ == '__main__':
    from math import isclose
    
    
    assert Vector(2) == Vector((0, 0, 1))
    Vector.rand(2)
    assert isclose(abs(Vector.randn(10)), 1)
    assert not isclose(abs(Vector.randn(10, normed=False)), 1)
    assert abs(Vector.ZERO) == 0
    
    v = Vector((1, 2, 3, 4, 5))
    
    assert len(v) == 5
    assert v[2] == 3
    assert v[999] == 0
    assert v[1:4:2] == Vector((2, 4))
    assert v[::].coef == (1, 2, 3, 4, 5)
    assert v[:6:].coef == (1, 2, 3, 4, 5, 0)
    assert Vector((1, 2, 3)) == Vector((1, 2, 3, 0))
    assert not Vector((1, 2, 3)) == Vector((1, 2, 3, 1))
    assert v<<1 == Vector((2, 3, 4, 5))
    assert v>>1 == Vector((0, 1, 2, 3, 4, 5))
    assert Vector((1, 0)).trim() == Vector((1,)) \
            and Vector(tuple()).trim() == Vector(tuple())
    
    assert isclose(abs(v), sqrt(55))
    assert v+Vector((3, 2, 1)) == Vector((4, 4, 4, 4, 5))
    assert v-Vector((3, 2, 1)) == Vector((-2, 0, 2, 4, 5))
    assert 2*v == v*2 == Vector((2, 4, 6, 8, 10))
    assert v/2 == Vector((0.5, 1.0, 1.5, 2, 2.5))
    assert v//2 == Vector((0, 1, 1, 2, 2))





import numpy as np
from itertools import zip_longest



#creation stuff
def vecnpzero(d=None):
    """Return `d` zero vectors.
    
    The retured value is a `(d, 1)`-array of zeros if `d` is not `None`
    or `[0]` otherwise.
    """
    #same dtype as numpy.polynomial.polynomial.polyzero
    return np.zeros(1 if d is None else (d, 1), dtype=np.int64)

def vecnpbasis(i, c=1, d=None):
    """Return `d` many `i`-th basis vectors times `c`.
    
    The retured value is a `(d, i+1)`-array if `d` is not `None`
    or `(i+1,)` otherwise.
    """
    #choose dtype acc to c
    v = np.zeros(i+1 if d is None else (d, i+1), dtype=np.asarray(c).dtype)
    v[..., -1] = c #maybe scalar, maybe (d,)-array
    return v

def vecnprand(n, d=None):
    """Return `d` random vectors of `n` uniform coefficients in `[0, 1[`.
    
    The retured value is a `(d, n)`-array if `d` is not `None`
    or `(n,)` otherwise.
    """
    return np.random.rand(*((n,) if d is None else (d, n)))

def vecnprandn(n, normed=True, d=None):
    """Return `d` random vectors of `n` normal distributed coefficients.
    
    The retured value is a `(d, n)`-array if `d` is not `None`
    or `(n,)` otherwise.
    """
    v = np.random.randn(*((n,) if d is None else (d, n)))
    return v/np.linalg.norm(v, axis=-1, keepdims=True) if normed else v


#sequence stuff
def vecnpeq(v, w):
    """Return if two vectors are equal."""
    v, w = np.asarray(v), np.asarray(w)
    if v.ndim == w.ndim == 1:
        return veceq(v, w)
    return np.array([veceq(v[i] if v.ndim>1 else v, w[i] if w.ndim>1 else w) \
            for i in range(v.shape[0] if v.ndim>1 else w.shape[0])])

def vecnptrim(v, tol=1e-9):
    """Remove all trailing near zero (abs(v_i)<=tol) coefficients."""
    v = np.asarray(v)
    #use np.all, because v[...,-1] maybe a scalar (not iterable)
    while v.shape[-1]>1 and np.all(abs(v[...,-1])<=tol):
        v = v[...,:-1]
    if v.shape[-1]==1 and np.all(abs(v[...,0])<=tol): #leave 'leading' zero
        v[...,0] = 0
    return v

#vecnpround = np.round


#Hilbert space stuff
def vecnpabsq(v):
    """Return the sum of absolute squares of the coefficients."""
    return np.sum(np.abs(v)**2, axis=-1)

def vecnpabs(v):
    """Return the Euclidean/L2-norm."""
    return np.linalg.norm(v, axis=-1)

def vecnpdot(v, w):
    """Return the inner product of two vectors without conjugation."""
    v, w = np.asarray(v), np.asarray(w)
    shape = tuple(reversed(tuple(
            map(min, zip(reversed(v.shape), reversed(w.shape))))))
    return np.sum(v[...,*map(slice, shape)]*w[...,*map(slice, shape)], axis=-1)


#vector space stuff
def vecnpadd(*vs):
    """Return the sum of vectors."""
    if not vs: #empty sum
        return vecnpzero()
    vs = tuple(np.asarray(v) for v in vs)
    shape = tuple(reversed(tuple(map(max, zip_longest( \
            *(reversed(v.shape) for v in vs), fillvalue=0)))))
    r = np.zeros(shape, dtype=np.result_type(*vs))
    for v in vs:
        r[...,*map(slice, v.shape)] += v
    return r

def vecnpsub(v, w):
    """Return the difference of two vectors."""
    v, w = np.asarray(v), np.asarray(w)
    shape = tuple(reversed(tuple(map(max, zip_longest( \
            reversed(v.shape), reversed(w.shape), fillvalue=0)))))
    r = np.zeros(shape, dtype=np.result_type(v, w))
    r[...,*map(slice, v.shape)] += v
    r[...,*map(slice, w.shape)] -= w
    return r



if __name__ == '__main__':
    from numpy.polynomial.polynomial import polyadd, polysub
    
    
    #creation
    #veczero
    assert np.array_equal(vecnpzero(), [0])
    assert np.array_equal(vecnpzero(1), [[0]])
    assert np.array_equal(vecnpzero(2), [[0],
                                         [0]])
    
    #vecbasis
    assert np.array_equal(vecnpbasis(2, 3), [0, 0, 3])
    assert np.array_equal(vecnpbasis(4, 3, 2), [[0, 0, 0, 0, 3],
                                                [0, 0, 0, 0, 3]])
    
    #vecrand
    assert vecnprand(2).shape == (2,)
    assert vecnprand(2, 3).shape == (3, 2)
    
    #vecgauss
    assert vecnprandn(2).shape == (2,)
    assert vecnprandn(2, d=3).shape == (3, 2)
    assert np.allclose(np.linalg.norm(vecnprandn(2, d=3), axis=1), [1, 1, 1])
    
    
    #veceq
    assert vecnpeq([1, 2], [1, 2, 0])
    assert not vecnpeq([1, 2], [1, 2, 1])
    assert np.array_equal(vecnpeq([[1, 2],
                                   [1, 3]], [1, 2, 0]), [True, False])
    assert np.array_equal(vecnpeq([1, 2], [[1, 2, 0],
                                           [1, 3, 1]]), [True, False])
    assert np.array_equal(vecnpeq([[1, 2],
                                   [3, 4]], [[1, 2, 0],
                                             [3, 4, 1]]), [True, False])
    
    #vectrim
    assert np.array_equal(vecnptrim([0, 0]), [0])
    assert np.array_equal(vecnptrim([0, 5, 1e-10]), [0, 5])
    assert np.array_equal(vecnptrim([[0, 0],
                                     [0, 0]]), [[0],
                                                [0]])
    assert np.array_equal(vecnptrim([[1, 2, 3e-10, 4e-12],
                                     [5, 6, 7    , 8e-13]]), [[1, 2, 3e-10],
                                                              [5, 6, 7]])
    
    #Hilbert
    #vecabsq
    assert vecnpabsq([1, 2]) == 5
    assert np.array_equal(vecnpabsq([[1, 2],
                                     [3, 4]]), [5, 25])
    
    #vecabs
    assert vecnpabs([3, 4]) == 5
    assert np.array_equal(vecnpabs([[3,  4],
                                    [5, 12]]), [5, 13])
    
    #vecdot
    assert vecnpdot([1, 2], [3, 4, 5]) == 11
    assert np.array_equal(vecnpdot([[1, 2],
                                    [3, 4]], [5, 6, 7]), [17, 39])
    assert np.array_equal(vecnpdot([1, 2], [[3, 4, 5],
                                            [6, 7, 8]]), [11, 20])
    assert np.array_equal(vecnpdot([[1, 2],
                                    [3, 4]], [[5, 6,  7],
                                              [8, 9, 10]]), [17, 60])
    
    #vector space
    #vecadd
    assert vecnpadd() == np.array([0])
    assert np.array_equal(vecnpadd([1, 2], [3, 4, 5]), [4, 6, 5])
    assert np.array_equal(vecnpadd([[1, 2],
                                    [3, 4]], [5, 6, 7]), [[6,  8, 7],
                                                          [8, 10, 7]])
    assert np.array_equal(vecnpadd([[1, 2],
                                    [3, 4]], [[5, 6,  7],
                                              [8, 9, 10]]), [[ 6,  8,  7],
                                                             [11, 13, 10]])
    for _ in range(100):
        #1D + 1D
        v = np.random.normal(size=np.random.randint(1, 20))
        w = np.random.normal(size=np.random.randint(1, 20))
        assert np.allclose(vecnpadd(v, w), polyadd(v, w))
        #1D + 2D
        w = np.random.normal(size=
                (np.random.randint(1, 20), np.random.randint(1, 20)))
        assert np.allclose(vecnpadd(v, w), [polyadd(v, wi) for wi in w])
        #2D + 2D
        v = np.random.normal(size=
                (np.random.randint(1, 20), np.random.randint(1, 20)))
        w = np.random.normal(size=(v.shape[0], np.random.randint(1, 20)))
        assert np.allclose(vecnpadd(v, w),
                [polyadd(vi, wi) for vi, wi in zip(v, w)])
    
    #vecsub
    assert np.array_equal(vecnpsub([1, 2], [3, 5, 7]), [-2, -3, -7])
    assert np.array_equal(vecnpsub([[1, 2],
                                    [3, 4]], [5, 7, 9]), [[-4, -5, -9],
                                                          [-2, -3, -9]])
    assert np.array_equal(vecnpsub([[1, 2],
                                    [3, 4]], [[5, 7,  9],
                                              [6, 8, 10]]), [[-4, -5,  -9],
                                                             [-3, -4, -10]])
    for _ in range(100):
        #1D + 1D
        v = np.random.normal(size=np.random.randint(1, 20))
        w = np.random.normal(size=np.random.randint(1, 20))
        assert np.allclose(vecnpsub(v, w), polysub(v, w))
        #1D + 2D
        w = np.random.normal(size=
                (np.random.randint(1, 20), np.random.randint(1, 20)))
        assert np.allclose(vecnpsub(v, w), [polysub(v, wi) for wi in w])
        #2D + 2D
        v = np.random.normal(size=
                (np.random.randint(1, 20), np.random.randint(1, 20)))
        w = np.random.normal(size=(v.shape[0], np.random.randint(1, 20)))
        assert np.allclose(vecnpsub(v, w),
                [polysub(vi, wi) for vi, wi in zip(v, w)])
