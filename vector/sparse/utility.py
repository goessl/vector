__all__ = ('vecslen',
           'vecseq',
           'vecstrim', 'vecsitrim',
           'vecsrshift', 'vecsirshift',
           'vecslshift', 'vecsilshift')



def vecslen(v):
    """Return the maximum set index plus one.
    
    Doesn't handle trailing zeros, use [`vecstrim`][vector.sparse.utility.vecstrim]
    if needed.
    """
    return max(v.keys(), default=0)

def vecseq(v, w):
    r"""Return whether two vectors are equal.
    
    $$
        \vec{v} \overset{?}{=} \vec{w}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be at most
    
    - $\min\{n, m\}$ scalar comparisons (`eq`) &
    - $|n-m|$ scalar boolean evaluations (`bool`).
    """
    for k in v.keys() | w.keys():
        if k not in w:
            if bool(v[k]):
                return False
        elif k not in v:
            if bool(w[k]):
                return False
        else:
            if not v[k]==w[k]:
                return False
    return True

def vecstrim(v, tol=None):
    r"""Remove all near zero (`abs(v_i)<=tol`) coefficients.
    
    $$
        \begin{pmatrix}
            v_0 \\
            \vdots \\
            v_m
        \end{pmatrix} \ \text{where} \ m=\max\{\, j\mid |v_j|>\text{tol}\,\}\cup\{-1\}
    $$
    
    `tol` may also be `None`,
    then all coefficients that evaluate to `False` are trimmed.
    
    Complexity
    ----------
    For a vector of $n$ elements there will be
    
    - $n$ scalar absolute evaluations (`abs`) &
    - $n$ scalar comparisons (`gt`).
    
    Notes
    -----
    - Cutting of elements that are `abs(v_i)<=tol` instead of `abs(v_i)<tol` to
    allow cutting of elements that are exactly zero by `trim(v, 0)` instead
    of `trim(v, sys.float_info.min)`.
    """
    if tol is None:
        return {i:vi for i, vi in v.items() if vi}
    else:
        return {i:vi for i, vi in v.items() if abs(vi)>tol}

def vecsitrim(v, tol=None):
    r"""Remove all near zero (`abs(v_i)<=tol`) coefficients.
    
    $$
        \begin{pmatrix}
            v_0 \\
            \vdots \\
            v_m
        \end{pmatrix} \ \text{where} \ m=\max\{\, j\mid |v_j|>\text{tol}\,\}\cup\{-1\}
    $$
    
    `tol` may also be `None`,
    then all coefficients that evaluate to `False` are trimmed.
    
    Complexity
    ----------
    For a vector of $n$ elements there will be
    
    - $n$ scalar absolute evaluations (`abs`) &
    - $n$ scalar comparisons (`gt`).
    
    Notes
    -----
    - Cutting of elements that are `abs(v_i)<=tol` instead of `abs(v_i)<tol` to
    allow cutting of elements that are exactly zero by `trim(v, 0)` instead
    of `trim(v, sys.float_info.min)`.
    """
    if tol is None:
        indices_to_del = tuple(i for i, vi in v.items() if not vi)
    else:
        indices_to_del = tuple(i for i, vi in v.items() if abs(vi)<=tol)
    
    for i in indices_to_del:
        del v[i]
    return v

def vecsrshift(v, n):
    r"""Shift coefficients up.
    
    $$
        (v_{i-n})_i \qquad \begin{pmatrix}
            0 \\
            \vdots \\
            0 \\
            v_0 \\
            v_1 \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{m+n}
    $$
    """
    return {i+n:vi for i, vi in v.items()}

def vecsirshift(v, n):
    r"""Shift coefficients up.
    
    $$
        (v_{i-n})_i \qquad \begin{pmatrix}
            0 \\
            \vdots \\
            0 \\
            v_0 \\
            v_1 \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{m+n}
    $$
    """
    for i in sorted(v.keys(), reverse=True):
        v[i+n] = v.pop(i)
    return v

def vecslshift(v, n):
    r"""Shift coefficients down.
    
    $$
        (v_{i+n})_i \qquad \begin{pmatrix}
            v_n \\
            v_{n+1} \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{\max\{m-n, 0\}}
    $$
    """
    return {i-n:vi for i, vi in v.items() if i-n>=0}

def vecsilshift(v, n):
    r"""Shift coefficients down.
    
    $$
        (v_{i+n})_i \qquad \begin{pmatrix}
            v_n \\
            v_{n+1} \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{\max\{m-n, 0\}}
    $$
    """
    for i in sorted(v.keys()):
        vi = v.pop(i)
        if i-n >= 0:
            v[i-n] = vi
    return v
