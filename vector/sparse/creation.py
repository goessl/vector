from random import random, gauss
from itertools import count
from .hilbertspace import vecsabs
from .vectorspace import vecstruediv
from typing import Any, Generator, Never
from collections.abc import Mapping



__all__ = ('vecszero', 'vecsbasis', 'vecsbases', 'vecsrand', 'vecsrandn')



vecszero:dict[Never,Never] = {}
r"""Zero vector.

$$
    \vec{0}
$$

An empty dictionary: `{}`.
"""

def vecsbasis(i:int, c:Any=1) -> dict[int,Any]:
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

def vecsbases(start:int=0, c:Any=1) -> Generator[dict[int,Any]]:
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

def vecsrand(n:int) -> dict[int,float]:
    r"""Return a random vector of uniformly sampled `float` coefficients.
    
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

def vecsrandn(n:int, normed:bool=True, mu:float=0, sigma:float=1, weights:Mapping[int,Any]|None=None) -> dict[int,Any]:
    r"""Return a random vector of normally sampled `float` coefficients.
    
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
