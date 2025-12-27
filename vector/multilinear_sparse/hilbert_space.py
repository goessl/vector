from ..lazy.hilbert_space import try_conjugate



__all__ = ('tensconj', )



def tensconj(t):
    r"""Return the complex.
    
    $$
        t^*
    $$
    
    Trys to call a method `conjugate` on each element.
    If not found, simply keeps the element as is.
    """
    return {i:try_conjugate(ti) for i, ti in t.items()}
