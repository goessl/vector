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
"""

def vecsbasis(i, c=1):
    r"""Return the `i`-th basis vector times `c`.
    
    $$
        c\vec{e}_i
    $$
    
    Returns a tuple with `i` many `zero`s followed by `c`.
    """
    return {i:c}

def vecsbases(start=0, c=1):
    r"""Yield all basis vectors.
    
    $$
        \left(\vec{e}_n\right)_\mathbb{n\in\mathbb{N_0}} = \left(\vec{e}_0, \vec{e}_1, \vec{e}_2, \dots \right)
    $$
    
    See also
    --------
    - for single basis vector: [`vecsbasis`][vector.sparse.creation.vecsbasis]
    """
    for i in count(start=start):
        yield vecsbasis(i, c=c)

def vecsrand(n):
    r"""Return a random vector of `n` uniform `float` coefficients in `[0, 1[`.
    
    $$
        \vec{v}\sim\mathcal{U}^n([0, 1[)
    $$
    """
    return {i:random() for i in range(n)}

def vecsrandn(n, normed=True, mu=0, sigma=1, weights=None):
    r"""Return a random vector of `n` normal distributed `float` coefficients.
    
    $$
        \vec{v}\sim\mathcal{N}^n(\mu, \sigma)
    $$
    """
    v = {i:gauss(mu, sigma) for i in range(n)}
    return vecstruediv(v, vecsabs(v, weights)) if normed else v
