import numpy as np
from itertools import zip_longest
from ._vecfunctions import veceq



__all__ = ['vecnpzero', 'vecnpbasis', 'vecnprand', 'vecnprandn',
        'vecnpdim', 'vecnpeq', 'vecnptrim', 'vecnpround',
        'vecnpabsq', 'vecnpabs', 'vecnpdot', 'vecnpparallel',
        'vecnppos', 'vecnpneg', 'vecnpadd', 'vecnpsub',
        'vecnpmul', 'vecnptruediv', 'vecnpfloordiv', 'vecnpmod']



#creation
def vecnpzero(d=None):
    """Return `d` zero vectors.
    
    The returned value is a `(d, 1)`-array of zeros if `d` is not `None`
    or `[0]` otherwise.
    """
    #same dtype as numpy.polynomial.polynomial.polyzero
    return np.zeros(1 if d is None else (d, 1), dtype=object)

def vecnpbasis(i, c=1, d=None):
    """Return `d` many `i`-th basis vectors times `c`.
    
    The returned value is a `(d, i+1)`-array if `d` is not `None`
    or `(i+1,)` otherwise.
    """
    #choose dtype acc to c
    v = np.zeros(i+1 if d is None else (d, i+1), dtype=np.dtype(type(c)))
    v[..., -1] = c #maybe scalar, maybe (d,)-array
    return v

def vecnprand(n, d=None):
    """Return `d` random vectors of `n` uniform coefficients in `[0, 1[`.
    
    The returned value is a `(d, n)`-array if `d` is not `None`
    or `(n,)` otherwise.
    """
    return np.random.rand(*((n,) if d is None else (d, n)))

def vecnprandn(n, normed=True, d=None):
    """Return `d` random vectors of `n` normal distributed coefficients.
    
    The returned value is a `(d, n)`-array if `d` is not `None`
    or `(n,)` otherwise.
    """
    v = np.random.randn(*((n,) if d is None else (d, n)))
    return v/np.linalg.norm(v, axis=-1, keepdims=True) if normed else v


#utility
def vecnpdim(v):
    """Return the number of allocated dimensions in this vector or vectors."""
    return np.asarray(v).shape[-1]

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
    while v.shape[-1]>1 and np.all(np.abs(v[...,-1])<=tol):
        v = v[...,:-1]
    if v.shape[-1]==1 and np.all(np.abs(v[...,0])<=tol): #leave 'leading' zero
        v[...,0] = 0
    return v

def vecnpround(v, ndigits=0):
    """Wrapper for `numpy.round`."""
    return np.round(np.asarray(v), ndigits)


#Hilbert space
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

def vecnpparallel(v, w):
    """Return if two vectors are parallel."""
    v, w = np.asarray(v), np.asarray(w)
    return vecnpabsq(v)*vecnpabsq(w) == vecnpdot(v, w)**2


#vector space
def vecnppos(v):
    """Return the vector with the unary positive operator applied."""
    return +np.asarray(v)

def vecnpneg(v):
    """Return the vector with the unary negative operator applied."""
    return -np.asarray(v)

def vecnpadd(*vs):
    """Return the sum of vectors."""
    if not vs: #empty sum
        return vecnpzero()
    
    vs = tuple(map(np.asarray, vs))
    if not all(v.ndim in {1, 2} for v in vs): #all 1D or 2D
        raise ValueError
    
    heights = {vdi for v in vs for vdi in v.shape[:-1]}
    if len(heights) > 1: #all 2D same height
        raise ValueError
    
    r = np.zeros(tuple(heights)+(max(v.shape[-1] for v in vs),),
            dtype=np.result_type(*vs))
    for v in vs:
        r[...,:v.shape[-1]] += v
    return r

def vecnpsub(v, w):
    """Return the difference of two vectors."""
    v, w = np.asarray(v), np.asarray(w)
    if v.ndim not in {1, 2} or w.ndim not in {1, 2}: #1D-1D, 1D-2D, 2D-1D, 2D-2D
        raise ValueError
    
    heights = set(v.shape[:-1]) | set(w.shape[:-1])
    if len(heights) > 1: #both same height if both 2D
        raise ValueError
    
    r = np.zeros(tuple(heights)+(max(v.shape[-1], w.shape[-1]),),
            dtype=np.result_type(v, w))
    r[...,:v.shape[-1]] += v
    r[...,:w.shape[-1]] -= w
    return r

def vecnpmul(a, v):
    """Return the product of a scalar and a vector."""
    return a * np.asarray(v)

def vecnptruediv(v, a):
    """Return the true division of a vector and a scalar."""
    return np.asarray(v) / a

def vecnpfloordiv(v, a):
    """Return the floor division of a vector and a scalar."""
    return np.asarray(v) // a

def vecnpmod(v, a):
    """Return the elementwise mod of a vector and a scalar."""
    return np.asarray(v) % a
