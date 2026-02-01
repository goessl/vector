from itertools import count
from ..lazy import veclrand, veclrandn
from .hilbertspace import vecabs
from .vectorspace import vectruediv



__all__ = ('veczero', 'vecbasis', 'vecbases', 'vecrand', 'vecrandn')



veczero = ()
r"""Zero vector.

$$
    \vec{0} \qquad \mathbb{K}^0
$$

An empty tuple: `()`.
"""

def vecbasis(i, c=1, zero=0):
    r"""Return a basis vector.
    
    $$
        c\vec{e}_i \qquad \mathbb{K}^{i+1}
    $$
    
    Returns a tuple with `i` many `zero`s followed by `c`.
    
    See also
    --------
    - for all basis vectors: [`vecbases`][vector.dense.creation.vecbases]
    """
    return (zero,)*i + (c,)

def vecbases(start=0, c=1, zero=0):
    r"""Yield all basis vectors.
    
    $$
        \left(\vec{e}_n\right)_{n\in\mathbb{N}_{\geq\text{start}}} = \left(\vec{e}_\text{start}, \vec{e}_{\text{start}+1}, \dots \right)
    $$
    
    See also
    --------
    - for single basis vector: [`vecbasis`][vector.dense.creation.vecbasis]
    """
    for i in count(start=start):
        yield vecbasis(i, c=c, zero=zero)

def vecrand(n):
    r"""Return a random vector of uniformly sampled `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{U}^n([0, 1[) \qquad \mathbb{K}^n
    $$
    
    The coefficients are sampled from a uniform distribution in `[0, 1[`.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    """
    return tuple(veclrand(n))

def vecrandn(n, normed=True, mu=0, sigma=1, weights=None):
    r"""Return a random vector of normally sampled `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{N}^n(\mu, \sigma) \qquad \mathbb{K}^n
    $$
    
    The coefficients are sampled from a normal distribution.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    """
    v = tuple(veclrandn(n, mu, sigma))
    return vectruediv(v, vecabs(v, weights)) if normed else v
