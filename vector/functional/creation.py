from itertools import count
from ..lazy import veclrand, veclrandn
from .hilbert_space import vecabs
from .vector_space import vectruediv




__all__ = ('veczero', 'vecbasis', 'vecbases', 'vecrand', 'vecrandn')



veczero = ()
r"""Zero vector.

$$
    \vec{0} \qquad \mathbb{K}^0
$$

An empty tuple: `()`.
"""

def vecbasis(i, c=1, zero=0):
    r"""Return the `i`-th basis vector times `c`.
    
    $$
        c\vec{e}_i \qquad \mathbb{K}^{i+1}
    $$
    
    Returns a tuple with `i` many `zero`s followed by `c`.
    """
    return (zero,)*i + (c,)

def vecbases(start=0, c=1, zero=0):
    r"""Yield all basis vectors.
    
    $$
        \left(\vec{e}_n\right)_{n\in\mathbb{N_0}} = \left(\vec{e}_0, \vec{e}_1, \vec{e}_2, \dots \right)
    $$
    
    See also
    --------
    - for single basis vector: [`vecbasis`][vector.functional.vecbasis]
    """
    for i in count(start=start):
        yield vecbasis(i, c=c, zero=zero)

def vecrand(n):
    r"""Return a random vector of `n` uniform `float` coefficients in `[0, 1[`.
    
    $$
        \vec{v}\sim\mathcal{U}^n([0, 1[) \qquad \mathbb{K}^n
    $$
    
    The coefficients are sampled from a uniform distribution in `[0, 1[`.
    
    Notes
    -----
    Naming like in `numpy.random`, because seems more concise
    (not `random` & `gauss` as in the stdlib).
    """
    return tuple(veclrand(n))

def vecrandn(n, normed=True, mu=0, sigma=1, weights=None):
    r"""Return a random vector of `n` normal distributed `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{N}^n(\mu, \sigma) \qquad \mathbb{K}^n
    $$
    
    The coefficients are sampled from a normal distribution.
    
    Notes
    -----
    Naming like in `numpy.random`, because seems more concise
    (not `random` & `gauss` as in the stdlib).
    """
    v = tuple(veclrandn(n, mu, sigma))
    return vectruediv(v, vecabs(v, weights)) if normed else v
