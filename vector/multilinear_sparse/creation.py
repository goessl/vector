from random import random, gauss
from ..functional.utility import vectrim
from numpy import ndindex



__all__ = ('tenszero', 'tensbasis', 'tensrand', 'tensrandn')



tenszero = {}
"""Zero tensor.

$$
    0
$$
"""

def tensbasis(i, c=1):
    """Return the `i`-th basis tensor times `c`.
    
    $$
        ce_i
    $$
    """
    return {vectrim(i):c}

def tensrand(*d):
    r"""Return a random tensor of `d` uniform coefficients in `[0, 1[`.
    
    $$
        t \sim \mathcal{U}^d([0, 1[)
    $$
    """
    return {vectrim(i):random() for i in ndindex(*d)}

def tensrandn(*d, mu=0, sigma=1):
    r"""Return a random tensor of `d` normal distributed coefficients.
    
    $$
        t \sim \mathcal{N}^d
    $$
    """
    return {vectrim(i):gauss(mu, sigma) for i in ndindex(*d)}
