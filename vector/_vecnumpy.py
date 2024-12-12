import numpy as np
from itertools import zip_longest
from ._vecfunctions import veceq



__all__ = ['vecnpzero', 'vecnpbasis', 'vecnprand', 'vecnprandn',
        'vecnpeq', 'vecnptrim',
        'vecnpabsq', 'vecnpabs', 'vecnpdot',
        'vecnpadd', 'vecnpsub']



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
    v = np.zeros(i+1 if d is None else (d, i+1), dtype=np.dtype(type(c)))
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
    while v.shape[-1]>1 and np.all(np.abs(v[...,-1])<=tol):
        v = v[...,:-1]
    if v.shape[-1]==1 and np.all(np.abs(v[...,0])<=tol): #leave 'leading' zero
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
    heights = set().union(*(v.shape[:-1] for v in vs))
    if len(heights) > 1:
        raise ValueError
    r = np.zeros(tuple(heights)+(max(v.shape[-1] for v in vs),),
            dtype=np.result_type(*vs))
    for v in vs:
        r[...,*map(slice, v.shape)] += v
    return r

def vecnpsub(v, w):
    """Return the difference of two vectors."""
    v, w = np.asarray(v), np.asarray(w)
    heights = set(v.shape[:-1] + w.shape[:-1])
    if len(heights) > 1:
        raise ValueError
    r = np.zeros(tuple(heights)+(max(v.shape[-1], w.shape[-1]),),
            dtype=np.result_type(v, w))
    r[...,*map(slice, v.shape)] += v
    r[...,*map(slice, w.shape)] -= w
    return r
