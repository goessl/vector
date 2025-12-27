__all__ = ('vecstod', 'vecdtos')



def vecstod(v, zero=0):
    """Return a sparse vector (`dict`) as a dense vector (`tuple`)."""
    d = [zero] * (max(v.keys(), default=-1)+1)
    for i, vi in v.items():
        d[i] = vi
    return tuple(d)

def vecdtos(v):
    """Return a dense vector (`tuple`) as a sparse vector (`dict`).
    
    The resulting `dict` is not [trimmed][vector.sparse.utility.vecstrim].
    """
    return {i:vi for i, vi in enumerate(v)}
