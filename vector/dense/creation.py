from itertools import chain, count, repeat
from ..lazy import veclrand, veclrandn
from .hilbertspace import vecabs
from .vectorspace import vectruediv
from typing import Any, Generator, TypeVar
from collections.abc import Callable, Iterable, Sequence



__all__ = ('veczero', 'vecbasis', 'vecbases', 'vecrand', 'vecrandn')



S = TypeVar('S', bound=Sequence)



class _VecZero(tuple):
    def __call__(self, factory:Callable[[],Sequence]=tuple) -> Sequence:
        return factory()

veczero:_VecZero = _VecZero()
r"""Zero vector.

$$
    \vec{0} \qquad \mathbb{K}^0
$$

An empty tuple that is also callable as a factory shorthand:
`veczero == ()`, `veczero()` returns `()`, `veczero(factory=list)` returns `[]`.
"""


def vecbasis(i:int, c:Any=1, zero:Any=0, factory:Callable[[Iterable],S]=tuple) -> S:
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

def vecbases(start:int=0, c:Any=1, zero:Any=0, factory:Callable[[Iterable],S]=tuple) -> Generator[S]:
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

def vecrand(n:int, factory:Callable[[Iterable],S]=tuple) -> S:
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
    return factory(veclrand(n))

def vecrandn(n:int, normed:bool=True, mu:float=0, sigma:float=1, weights:Iterable|None=None, factory:Callable[[Iterable],S]=tuple) -> S:
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
    if not normed:
        return factory(veclrandn(n, mu, sigma))
    else:
        v:tuple[float,...] = tuple(veclrandn(n, mu, sigma))
        return vectruediv(v, vecabs(v, weights), factory=factory)
