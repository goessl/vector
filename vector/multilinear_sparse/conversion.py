from ..functional.utility import vectrim
from .utility import tensdim
import numpy as np



__all__ = ('tenstod', 'tendtos')



def tenstod(t, zero=0):
    """Return a sparse tensor (`dict`) as a dense tensor (`numpy.ndarray`)."""
    r = np.full(tensdim(t), zero, dtype=np.array(t.values).dtype)
    for i, ti in t.items():
        r[i] = ti
    return r

def tendtos(t):
    """Return a dense tensor (`numpy.ndarray`) as a sparse tensor (`dict`).
    
    The resulting `dict` is not [trimmed][vector.multilinear_sparse.tenstrim].
    """
    return {vectrim(i):ti for i, ti in np.ndenumerate(t)}
