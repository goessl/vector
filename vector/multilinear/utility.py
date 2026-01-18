import numpy as np



__all__ = ('tenrank', 'tendim', 'teneq', 'tentrim', 'tenrshift', 'tenlshift')



def tenrank(t):
    r"""Return the rank.
    
    $$
        \text{rank}\,t
    $$
    
    See also
    --------
    - wraps: [`numpy.ndarray.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html)
    """
    return np.asarray(t).ndim

def tendim(t):
    r"""Return the dimensionalities.
    
    $$
        \dim t
    $$
    
    See also
    --------
    - wraps: [`numpy.ndarray.shape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html)
    """
    return np.asarray(t).shape

def teneq(s, t):
    r"""Return whether two tensors are equal.
    
    $$
        s\overset{?}{=}t
    $$
    """
    raise NotImplementedError

def tentrim(t, tol=None):
    """Remove all trailing near zero (`abs(t_i)<=tol`) coefficients.
    
    `tol` may also be `None`,
    then all coefficients that evaluate to `False` are trimmed.
    
    Notes
    -----
    - Cutting of elements that are `abs(t_i)<=tol` instead of `abs(t_i)<tol` to
    allow cutting of elements that are exactly zero by `trim(t, 0)` instead
    of `trim(t, sys.float_info.min)`.
    """
    t = np.asarray(t)
    for d in range(t.ndim): #reduce dimension
        slc_idx = (slice(None),)*d + (-1,) + (...,)
        slc_drop = (slice(None),)*d + (slice(-1),) + (...,)
        while t.shape[d]>0 and np.all(np.logical_not(t[slc_idx].astype(bool)) if tol is None else np.abs(t[slc_idx])<=tol):
            t = t[slc_drop]
    if t.size == 0:
        return t.reshape((0,))
    while t.ndim>1 and t.shape[-1]==1: #reduce rank
        t = t[..., 0]
    return t

def tenrshift(t, n):
    """Shift coefficients up.
    
    See also
    --------
    - wraps: [`numpy.pad`](https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
    """
    return np.pad(t, tuple((ni, 0) for ni in n))

def tenlshift(t, n):
    """Shift coefficients down.
    
    See also
    --------
    - wraps: [`numpy.pad`](https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
    """
    return np.array(t)[*(slice(ni, None) for ni in n)].copy()
