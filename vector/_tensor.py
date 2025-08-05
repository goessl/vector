import numpy as np
from ._vecfunctions import vechadamardmax



__all__ = ['tenzero', 'tenbasis', 'tenrand', 'tenrandn',
        'tenrank', 'tendim', 'tentrim', 'tenround',
        'tenpos', 'tenneg', 'tenaddc', 'tenadd', 'tensub', 'tenmul',
        'tentruediv', 'tenfloordiv', 'tenmod',
        'tenhadamard', 'tenhadamardtruediv',
        'tenhadamardfloordiv', 'tenhadamardmod']



#creation
tenzero = np.zeros((), dtype=object)
"""Zero tensor."""
tenzero.flags.writeable = False

def tenbasis(i, c=1):
    """Return the `i`-th basis tensor times `c`."""
    t = np.zeros(np.add(i, 1), dtype=np.result_type(c))
    t[i] = c #dont unpack i, it might be a scalar
    return t

def tenrand(*d):
    """Wrapper for `numpy.random.rand`."""
    return np.random.rand(*d)

def tenrandn(*d):
    """Wrapper for `numpy.random.randn`."""
    return np.random.randn(*d)


#utility stuff
def tenrank(t):
    """Return the rank of the tensor."""
    return np.asarray(t).ndim

def tendim(t):
    """Return the dimensionalities of the tensor."""
    return np.asarray(t).shape

def tentrim(t, tol=1e-9):
    """Remove all trailing near zero (abs(v_i)<=tol) coefficients."""
    t = np.asarray(t)
    for d in range(t.ndim):
        i = (slice(None, None, None),)*d + (-1,) + (...,)
        while t.shape[d]>0 and np.all(np.abs(t[*i])<=tol):
            t = t[(slice(None, None, None),)*d + (slice(0, -1),) + (...,)]
    return t

def tenround(t, ndigits=0):
    """Wrapper for `numpy.round`."""
    return np.round(t, ndigits)


#vector space stuff
def tenpos(t):
    """Return the tensor with the unary positive operator applied."""
    return +np.asarray(t)

def tenneg(t):
    """Return the tensor with the unary negative operator applied."""
    return -np.asarray(t)

def tenaddc(t, c, i=(0,)):
    """Return `t` with `c` added to the `i`-th coefficient.
    
    More efficient than `tenadd(v, tenbasis(i, c)`.
    """
    t = np.asarray(t)
    while t.ndim < len(i):
        t = np.expand_dims(t, axis=-1)
    t = np.pad(t, tuple((0, max(ii-s+1, 0)) for s, ii in zip(t.shape, i)))
    t[i + (0,)*(len(i)-t.ndim)] += c
    return t

def tenadd(*ts):
    """Return the sum of tensors."""
    ts = tuple(map(np.asarray, ts))
    shape = vechadamardmax(*(t.shape for t in ts))
    r = np.zeros(shape, dtype=np.result_type(*ts) if ts else object)
    for t in ts:
        r[tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)] += t
    return r

def tensub(s, t):
    """Return the difference of two tensors."""
    s, t = np.asarray(s), np.asarray(t)
    shape = vechadamardmax(s.shape, t.shape)
    r = np.zeros(shape, dtype=np.result_type(s, t))
    r[tuple(map(slice, s.shape)) + (0,)*(r.ndim-s.ndim)] = s
    r[tuple(map(slice, t.shape)) + (0,)*(r.ndim-t.ndim)] -= t
    return r

def tenmul(a, t):
    """Return the product of a scalar and a tensor."""
    return a * np.asarray(t)

def tentruediv(t, a):
    """Return the true division of a tensor and a scalar."""
    return np.asarray(t) / a

def tenfloordiv(t, a):
    """Return the floor division of a tensor and a scalar."""
    return np.asarray(t) // a

def tenmod(t, a):
    """Return the elementwise mod of a tensor and a scalar."""
    return np.asarray(t) % a


#elementwise operations
def tenhadamard(*ts):
    """Return the elementwise product of tensors."""
    ts = tuple(map(np.asarray, ts))
    shape = tuple(map(min, zip(*(t.shape for t in ts))))
    r = np.zeros(shape, dtype=np.result_type(*ts) if ts else object)
    if ts:
        r = ts[0][tuple(map(slice, shape)), ...]
    for t in ts[1:]:
        r *= t[tuple(map(slice, shape)), ...]
    return r

def tenhadamardtruediv(s, t):
    """Return the elementwise true division of two tensors."""
    s, t = np.asarray(s), np.asarray(t)
    shape = tuple(map(min, zip(s.shape, t.shape)))
    r = np.zeros(shape, dtype=np.result_type(s, t))
    r = s[tuple(map(slice, shape)), ...]
    r /= t[tuple(map(slice, shape)), ...]
    return r

def tenhadamardfloordiv(s, t):
    """Return the elementwise floor division of two tensors."""
    s, t = np.asarray(s), np.asarray(t)
    shape = tuple(map(min, zip(s.shape, t.shape)))
    r = np.zeros(shape, dtype=np.result_type(s, t))
    r = s[tuple(map(slice, shape)), ...]
    r //= t[tuple(map(slice, shape)), ...]
    return r

def tenhadamardmod(s, t):
    """Return the elementwise mod of two tensors."""
    s, t = np.asarray(s), np.asarray(t)
    shape = tuple(map(min, zip(s.shape, t.shape)))
    r = np.zeros(shape, dtype=np.result_type(s, t))
    r = s[tuple(map(slice, shape)), ...]
    r %= t[tuple(map(slice, shape)), ...]
    return r
