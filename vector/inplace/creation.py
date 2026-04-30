from itertools import count
from ..lazy.creation import veclrand, veclrandn
from ..dense.hilbertspace import vecabs
from .vectorspace import vecitruediv
from typing import Any, Generator, Never
from collections.abc import Iterable



__all__ = ('vecizero', 'vecibasis', 'vecibases', 'vecirand', 'vecirandn')



def vecizero() -> list[Never]:
    r"""Return a zero vector.
    
    $$
        \vec{0} \qquad \mathbb{K}^0
    $$
    
    An empty list: `[]`.
    """
    return []

def vecibasis(i:int, c:Any=1, zero:Any=0) -> list[Any]:
    r"""Return a basis vector.
    
    $$
        c\vec{e}_i \qquad \mathbb{K}^{i+1}
    $$
    
    Returns a list with `i` many `zero`s followed by `c`.
    
    See also
    --------
    - for all basis vectors: [`vecibases`][vector.inplace.creation.vecibases]
    """
    return [zero]*i + [c]

def vecibases(start:int=0, c:Any=1, zero:Any=0) -> Generator[list[Any]]:
    r"""Yield all basis vectors.
    
    $$
        \left(\vec{e}_n\right)_{n\in\mathbb{N}_{\geq\text{start}}} = \left(\vec{e}_\text{start}, \vec{e}_{\text{start}+1}, \dots \right)
    $$
    
    See also
    --------
    - for single basis vector: [`vecibasis`][vector.inplace.creation.vecibasis]
    """
    for i in count(start=start):
        yield vecibasis(i, c=c, zero=zero)

def vecirand(n:int) -> list[float]:
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
    return list(veclrand(n))

def vecirandn(n:int, normed:bool=True, mu:float=0, sigma:float=1, weights:Iterable[Any]|None=None) -> list[Any]:
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
    v = list(veclrandn(n, mu, sigma))
    return vecitruediv(v, vecabs(v, weights)) if normed else v
