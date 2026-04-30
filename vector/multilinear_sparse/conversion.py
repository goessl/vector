from ..dense.utility import vectrim
from .utility import tensdim
import numpy as np
from typing import Any
from collections.abc import Mapping



__all__ = ('tenstod', 'tendtos')



def tenstod(t:Mapping[tuple[int,...],Any], zero:Any=0) -> np.ndarray:
    """Return a sparse tensor (`dict`) as a dense tensor (`numpy.ndarray`)."""
    r = np.full(tensdim(t), zero, dtype=np.array(t.values).dtype)
    for i, ti in t.items():
        r[i] = ti
    return r

def tendtos(t:np.ndarray) -> dict[tuple[int,...],Any]:
    """Return a dense tensor (`numpy.ndarray`) as a sparse tensor (`dict`).
    
    The resulting `dict` is not [trimmed][vector.multilinear_sparse.tenstrim].
    """
    return {vectrim(i):ti for i, ti in np.ndenumerate(t)}
