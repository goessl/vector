from random import random, gauss
from itertools import count
from .hilbert_space import vecsabs
from .vector_space import vecstruediv




__all__ = ('vecszero', 'vecsbasis', 'vecsbases', 'vecsrand', 'vecsrandn')



vecszero = {}
r"""Zero vector.

$$
    \vec{0}
$$

An empty dictionary: `{}`.
"""

def vecsbasis(i, c=1):
    r"""Return a basis vector.
    
    $$
        c\vec{e}_i
    $$
    
    Returns a dictionary with a single element `i:c`.
    
    See also
    --------
    - for all basis vectors: [`vecsbases`][vector.sparse.creation.vecsbases]
    """
    return {i:c}

def vecsbases(start=0, c=1):
    r"""Yield all basis vectors.
    
    $$
        \left(\vec{e}_n\right)_{n\in\mathbb{N}_{\geq\text{start}}} = \left(\vec{e}_\text{start}, \vec{e}_{\text{start}+1}, \dots \right)
    $$
    
    See also
    --------
    - for single basis vector: [`vecsbasis`][vector.sparse.creation.vecsbasis]
    """
    for i in count(start=start):
        yield vecsbasis(i, c=c)

def vecsrand(n):
    r"""Return a random vector of uniform sampled `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{U}^n([0, 1[)
    $$
    
    The coefficients are sampled from a uniform distribution in `[0, 1[`.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    """
    return {i:random() for i in range(n)}

def vecsrandn(n, normed=True, mu=0, sigma=1, weights=None):
    r"""Return a random vector of normal sampled `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{N}^n(\mu, \sigma)
    $$
    
    The coefficients are sampled from a normal distribution.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    """
    v = {i:gauss(mu, sigma) for i in range(n)}
    return vecstruediv(v, vecsabs(v, weights)) if normed else v
