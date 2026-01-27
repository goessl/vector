import numpy as np



__all__ = ('tenconj', 'tenprod')



def tenconj(t):
    """Return the elementwise complex conjugate.
    
    $$
        t^*
    $$
    
    See also
    --------
    - one-dimensional: [`vecconj`][vector.dense.hilbert_space.vecconj]
    - wraps: [`numpy.conjugate`](https://numpy.org/doc/stable/reference/generated/numpy.conjugate.html)
    """
    return np.conjugate(t)

def tenprod(s, t):
    r"""Return the tensor product.
    
    $$
        s \otimes t
    $$
    
    See also
    --------
    - wraps: [`numpy.tensordot](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html)
    """
    return np.tensordot(s, t, 0)
