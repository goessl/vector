import numpy as np



__all__ = ('tenrank', 'tendim', 'tentrim', 'tenround')



def tenrank(t):
    r"""Return the rank of a tensor.
    
    $$
        \text{rank}\,t
    $$
    
    See also
    --------
    - wraps: [`numpy.ndarray.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html)
    """
    return np.asarray(t).ndim

def tendim(t):
    r"""Return the dimensionalities of a tensor.
    
    $$
        \dim t
    $$
    
    See also
    --------
    - one-dimensional: [`veclen`][vector.functional.utility.veclen]
    - wraps: [`numpy.ndarray.shape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html)
    """
    return np.asarray(t).shape

#TODO: teneq

def tentrim(t, tol=1e-9):
    """Remove all trailing near zero (`abs(v_i)<=tol`) coefficients.
    
    See also
    --------
    - one-dimensional: [`vectrim`][vector.functional.utility.vectrim]
    """
    t = np.asarray(t)
    for d in range(t.ndim): #reduce dimension
        slc = (slice(None),)*d + (-1,) + (...,)
        while t.shape[d]>0 and np.all(np.abs(t[*slc])<=tol):
            t = t[(slice(None),)*d + (slice(-1),) + (...,)]
    if t.size == 0:
        return t.reshape((0,))
    while t.shape and t.shape[-1] == 1: #reduce rank
        t = t[..., 0]
    return t

def tenround(t, ndigits=0):
    r"""Round all coefficients to the given precision.
    
    $$
        (\text{round}_\text{ndigits}(v_i))_i
    $$
    
    See also
    --------
    - one-dimensional: [`vecround`][vector.functional.utility.vecround]
    - wraps: [`numpy.round`](https://numpy.org/doc/stable/reference/generated/numpy.round.html)
    """
    return np.round(t, decimals=ndigits)

def tenrshift(t, n):
    r"""Pad `n` many zeros to the beginning of the tensor.
    
    See also
    --------
    - one-dimensional: [`vecrshift`][vector.functional.utility.vecrshift]
    - wraps: [`numpy.pad`](https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
    """
    return np.pad(t, tuple((ni, 0) for ni in n))

def tenlshift(t, n):
    r"""Remove `n` many coefficients at the beginning of the tensor.
    
    See also
    --------
    - one-dimensional: [`veclshift`][vector.functional.utility.veclshift]
    - wraps: [`numpy.pad`](https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
    """
    return np.array(t)[*(slice(ni, None) for ni in n)].copy()
