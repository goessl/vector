from ..util import try_conjugate
from typing import Any
from collections.abc import Mapping, MutableMapping



__all__ = ('tensconj', 'tensiconj')



def tensconj(t:Mapping[tuple[int,...],Any]) -> dict[tuple[int,...],Any]:
    """Return the complex conjugate.
    
    $$
        t^*
    $$
    
    Tries to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    """
    return {i:try_conjugate(ti) for i, ti in t.items()}

def tensiconj(t:MutableMapping[tuple[int,...],Any]) -> MutableMapping[tuple[int,...],Any]:
    """Complex conjugate.
    
    $$
        t = t^*
    $$
    
    Tries to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    """
    for i, ti in t.values():
        t[i] = try_conjugate(ti)
    return t
