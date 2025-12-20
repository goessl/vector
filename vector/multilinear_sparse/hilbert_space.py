from ..lazy.hilbert_space import try_conjugate



__all__ = ('tensconj', )



def tensconj(t):
    r"""Return the elementwise complex conjugate.
    
    $$
        t^*
    $$
    """
    return {i:try_conjugate(ti) for i, ti in t.items()}
