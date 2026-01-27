from random import random, gauss
from ..dense.utility import vectrim
from numpy import ndindex



__all__ = ('tenszero', 'tensbasis', 'tensrand', 'tensrandn')



tenszero = {}
"""Zero tensor.

$$
    0
$$

An empty dictionary: `{}`.
"""

def tensbasis(i, c=1):
    """Return a basis tensor.
    
    $$
        ce_i
    $$
    
    Returns a dictionary with a single element `i:c`.
    """
    return {vectrim(i):c}

def tensrand(*d):
    r"""Return a random tensor of uniform sampled `float` coefficients.
    
    $$
        t \sim \mathcal{U}^d([0, 1[)
    $$
    
    The coefficients are sampled from a uniform distribution in `[0, 1[`.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    """
    return {vectrim(i):random() for i in ndindex(*d)}

def tensrandn(*d, mu=0, sigma=1):
    r"""Return a random tensor of normal sampled `float` coefficients.
    
    $$
        t \sim \mathcal{N}^d(\mu, \sigma)
    $$
    
    The coefficients are sampled from a normal distribution.
    
    Notes
    -----
    Naming like [`numpy.random`](https://numpy.org/doc/stable/reference/random/legacy.html),
    because seems more concise (not `random` & `gauss` as in the stdlib).
    """
    return {vectrim(i):gauss(mu, sigma) for i in ndindex(*d)}
