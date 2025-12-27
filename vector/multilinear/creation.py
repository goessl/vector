import numpy as np



__all__ = ('tenzero', 'tenbasis', 'tenrand', 'tenrandn')



tenzero = np.zeros((), dtype=object)
r"""Zero tensor.

$$
    0 \qquad \mathbb{K}^0
$$

An empty array.

Notes
-----
Why shape `(0,)` (=one dimensional, zero length) instead of `()` (zero dimensional)?

Shape `()` would be size one (empty product) and a scalar that could have any nonzero value.

Dimensionality of one isn't perfect, but at least its size is then zero and it couldn't be any arbitrary value.
"""
tenzero.flags.writeable = False

def tenbasis(i, c=1):
    """Return a basis tensor.
    
    $$
        ce_i
    $$
    
    Returns a `numpy.ndarray` with `i+1` zeros in each direction and a `c` in
    the outer corner.
    """
    t = np.zeros(np.add(i, 1), dtype=np.result_type(c))
    t[i] = c #dont unpack i, it might be a scalar
    return t

def tenrand(*d):
    r"""Return a random tensor of uniform sampled `float` coefficients.
    
    $$
        t \sim \mathcal{U}^d([0, 1[)
    $$
    
    The coefficients are sampled from a uniform distribution in `[0, 1[`.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    
    See also
    --------
    - wraps: [`numpy.random.rand`](https://numpy.org/doc/stable/reference/generated/numpy.random.rand.html)
    """
    return np.random.rand(*d)

def tenrandn(*d):
    r"""Return a random tensor of normal sampled `float` coefficients.
    
    $$
        t \sim \mathcal{N}^d(0, 1)
    $$
    
    The coefficients are sampled from a normal distribution.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    
    See also
    --------
    - wraps: [`numpy.random.randn`](https://numpy.org/doc/stable/reference/generated/numpy.random.randn.html)
    """
    return np.random.randn(*d)
