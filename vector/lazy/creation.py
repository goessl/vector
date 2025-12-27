from random import random, gauss
from itertools import chain, count, repeat



__all__ = ('veclzero', 'veclbasis', 'veclbases', 'veclrand', 'veclrandn')



def veclzero():
    r"""Zero vector.
    
    $$
        \vec{0} \qquad \mathbb{K}^0
    $$
    
    An empty generator.
    """
    yield from ()

def veclbasis(i, c=1, zero=0):
    r"""Return a basis vector.
    
    $$
        c\vec{e}_i \qquad \mathbb{K}^{i+1}
    $$
    
    Yields `i` zeros followed by `c`.
    
    See also
    --------
    - for all basis vectors: [`veclbases`][vector.lazy.creation.veclbases]
    """
    yield from chain(repeat(zero, i), (c,))

def veclbases(start=0, c=1, zero=0):
    r"""Yield all basis vectors.
    
    $$
        \left(\vec{e}_n\right)_{n\in\mathbb{N}_{\geq\text{start}}} = \left(\vec{e}_\text{start}, \vec{e}_{\text{start}+1}, \dots \right)
    $$
    
    See also
    --------
    - for single basis vector: [`veclbasis`][vector.lazy.creation.veclbases]
    """
    for i in count(start=start):
        yield veclbasis(i, c=c, zero=zero)

def veclrand(n):
    r"""Return a random vector of uniform sampled `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{U}^n([0, 1[) \qquad \mathbb{K}^n
    $$
    
    The coefficients are sampled from a uniform distribution in `[0, 1[`.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    """
    yield from (random() for _ in range(n))

def veclrandn(n, mu=0, sigma=1):
    r"""Return a random vector of normal sampled `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{N}^n(\mu, \sigma) \qquad \mathbb{K}^n
    $$
    
    The coefficients are sampled from a normal distribution.
    
    Difference to [`vecrandn`][vector.functional.vecrandn]:
    The vector can't be normalised as it isn't materialised.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    """
    yield from (gauss(mu, sigma) for _ in range(n))
