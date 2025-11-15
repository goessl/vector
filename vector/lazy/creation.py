from random import random, gauss
from itertools import chain, count, repeat



__all__ = ('veclzero', 'veclbasis', 'veclbases', 'veclrand', 'veclrandn')



def veclzero():
    r"""Zero vector.
    
    $$
        \vec{0} \qquad \mathbb{K}^0
    $$
    """
    yield from ()

def veclbasis(i, c=1, zero=0):
    r"""Return the `i`-th basis vector times `c`.
    
    $$
        c\vec{e}_i \qquad \mathbb{K}^{i+1}
    $$
    
    Returns a tuple with `i` zeros followed by `c`.
    """
    yield from chain(repeat(zero, i), (c,))

def veclbases(start=0, c=1, zero=0):
    r"""Yield all basis vectors.
    
    $$
        \left(\vec{e}_n\right)_\mathbb{n\in\mathbb{N_0}} = \left(\vec{e}_0, \vec{e}_1, \vec{e}_2, \dots \right)
    $$
    
    See also
    --------
    - for single basis vector: [`vecbasis`][vector.functional.vecbasis]
    """
    for i in count(start=start):
        yield veclbasis(i, c=c, zero=zero)

def veclrand(n):
    r"""Return a random vector of `n` uniform `float` coefficients in `[0, 1[`.
    
    $$
        \vec{v}\sim\mathcal{U}^n([0, 1[) \qquad \mathbb{K}^n
    $$
    
    Notes
    -----
    Naming like in `numpy.random`, because seems more concise
    (not `random` & `gauss` as in the stdlib).
    """
    yield from (random() for _ in range(n))

def veclrandn(n, mu=0, sigma=1):
    r"""Return a random vector of `n` normal distributed `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{N}^n(\mu, \sigma) \qquad \mathbb{K}^n
    $$
    
    Difference to [`vecrandn`][vector.functional.vecrandn]: The vector can't be normalised as it isn't materialised.
    
    Notes
    -----
    Naming like in `numpy.random`, because seems more concise
    (not `random` & `gauss` as in the stdlib).
    """
    yield from (gauss(mu, sigma) for _ in range(n))
