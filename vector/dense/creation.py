from random import random, gauss
from itertools import chain, count, repeat
from .hilbertspace import vecabs
from .vectorspace import vectruediv
from typing import Generator, TypeVar
from collections.abc import Callable, Iterable



__all__ = ('veczero',
           'vecbasis', 'vecbases',
           'vecrand',  'vecrandn')



T = TypeVar('T')
V = TypeVar('V')



class _VecZero(tuple):
    def __call__(self, factory:Callable[[tuple[()]],V]=tuple) -> V:
        return factory(self)

veczero:_VecZero = _VecZero()
r"""Zero vector.

$$
    \vec{0} \qquad \mathbb{K}^0
$$

An empty tuple that is also callable as a factory shorthand:
`veczero == ()`, `veczero()` returns `()`, `veczero(factory=list)` returns `[]`.
"""


def vecbasis(i:int, c:T=1, zero:T=0, factory:Callable[[Iterable[T]],V]=tuple) -> V:
    r"""Return a basis vector.
    
    $$
        c\vec{e}_i \qquad \mathbb{K}^{i+1}
    $$
    
    Returns a tuple with `i` many `zero`s followed by `c`.
    
    See also
    --------
    - for all basis vectors: [`vecbases`][vector.dense.creation.vecbases]
    """
    return factory(chain(repeat(zero, i), (c,)))

def vecbases(start:int=0, c:T=1, zero:T=0, factory:Callable[[Iterable[T]],V]=tuple) -> Generator[V]:
    r"""Yield all basis vectors.
    
    $$
        \left(\vec{e}_n\right)_{n\in\mathbb{N}_{\geq\text{start}}} = \left(\vec{e}_\text{start}, \vec{e}_{\text{start}+1}, \dots \right)
    $$
    
    See also
    --------
    - for single basis vector: [`vecbasis`][vector.dense.creation.vecbasis]
    """
    for i in count(start=start):
        yield vecbasis(i, c=c, zero=zero, factory=factory)

def vecrand(n:int, factory:Callable[[Iterable[float]],V]=tuple) -> V:
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
    return factory(random() for _ in range(n))

def vecrandn(n:int, normed:bool=True, mu:float=0.0, sigma:float=1.0, weights:Iterable[float]|None=None, factory:Callable[[Iterable[float]],V]=tuple) -> V:
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
    result = (gauss(mu, sigma) for _ in range(n))
    if not normed:
        return factory(result)
    else:
        v:tuple[float,...] = tuple(result)
        return vectruediv(v, vecabs(v, weights), factory=factory)
