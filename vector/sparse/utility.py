__all__ = ('vecseq', 'vecstrim', 'vecsrshift', 'vecslshift')



def vecseq(v, w):
    r"""Return if two vectors are equal.
    
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

def vecstrim(v, tol=1e-9):
    r"""Remove all near zero (`abs(v_i)<=tol`) coefficients.
    
    $$
        \begin{pmatrix}
            v_0 \\
            \vdots \\
            v_m
        \end{pmatrix} \ \text{where} \ m=\max\{\, j\mid |v_j|>\text{tol}\,\}\cup\{-1\}
    $$
    
    Complexity
    ----------
    For a vector of $n$ elements there will be
    
    - $n$ scalar absolute evaluations (`abs`) &
    - $n$ scalar comparisons (`gt`).
    
    Notes
    -----
    - Cutting of elements that are `abs(vi)<=tol` instead of `abs(vi)<tol` to
    allow cutting of elements that are exactly zero by `trim(v, 0)` instead
    of `trim(v, sys.float_info.min)`.
    - `tol=1e-9` like in [PEP 485](https://peps.python.org/pep-0485/#defaults).
    """
    return {i:vi for i, vi in v.items() if (vi if tol is None else abs(vi)>tol)}

def vecsrshift(v, n):
    r"""Pad `n` many `zero`s to the beginning of the vector.
    
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

def vecslshift(v, n):
    r"""Remove `n` many coefficients at the beginning of the vector.
    
    $$
        (v_{i+n})_i \qquad \begin{pmatrix}
            v_n \\
            v_{n+1} \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{\max\{m-n, 0\}}
    $$
    """
    return {i-n:vi for i, vi in v.items() if i-n>=0}
