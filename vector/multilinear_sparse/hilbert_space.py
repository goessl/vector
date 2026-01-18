from ..lazy.hilbert_space import try_conjugate



__all__ = ('tensconj', 'tensiconj')



def tensconj(t):
    """Return the complex conjugate.
    
    $$
        t^*
    $$
    
    Tries to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    """
    return {i:try_conjugate(ti) for i, ti in t.items()}

def tensiconj(t):
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
