import numpy as np



__all__ = ('tenzero', 'tenbasis', 'tenrand', 'tenrandn')



tenzero = np.zeros((), dtype=object)
"""Zero tensor.

$$
    0
$$

Notes
-----
Why shape `(0,)` (=one dimensional, zero length) instead of `()` (zero dimensional)?

Shape `()` would be size one (empty product) and a scalar that could have any nonzero value.

Dimensionality of one isn't perfect, but at least its size is then zero and it couldn't be any arbitrary value.

See also
--------
- one-dimensional: [`veczero`][vector.functional.veczero]
"""
tenzero.flags.writeable = False

def tenbasis(i, c=1):
    """Return the `i`-th basis tensor times `c`.
    
    $$
        ce_i
    $$
    
    Returns a `numpy.ndarray` with `i+1` zeros in each direction and a `c` in
    the outer corner.
    
    See also
    --------
    - one-dimensional: [`vecbasis`][vector.functional.creation.vecbasis]
    """
    t = np.zeros(np.add(i, 1), dtype=np.result_type(c))
    t[i] = c #dont unpack i, it might be a scalar
    return t

def tenrand(*d):
    r"""Return a random tensor of `d` uniform coefficients in `[0, 1[`.
    
    $$
        t \sim \mathcal{U}^d([0, 1[)
    $$
    
    See also
    --------
    - one-dimensional: [`vecrand`][vector.functional.creation.vecrand]
    - wraps: [`numpy.random.rand`](https://numpy.org/doc/stable/reference/generated/numpy.random.rand.html)
    """
    return np.random.rand(*d)

def tenrandn(*d):
    r"""Return a random tensor of `d` normal distributed coefficients.
    
    $$
        t \sim \mathcal{N}^d
    $$
    
    See also
    --------
    - one-dimensional: [`vecrandn`][vector.functional.creation.vecrandn]
    - wraps: [`numpy.random.randn`](https://numpy.org/doc/stable/reference/generated/numpy.random.randn.html)
    """
    return np.random.randn(*d)
