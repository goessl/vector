from ..lazy import vecleq, veclrshift, vecllshift



__all__ = ('veclen', 'veceq', 'vectrim', 'vecrshift', 'veclshift')



def veclen(v):
    """Return the length (number of set coefficients).
    
    Doesn't handle trailing zeros, use [`vectrim`][vector.functional.utility.vectrim]
    if needed.
    
    Notes
    -----
    For generators as they have no `len`gth, altough the vector is gone then.
    """
    return sum(1 for _ in v)

def veceq(v, w):
    r"""Return whether two vectors are equal.
    
    $$
        \vec{v}\overset{?}{=}\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{B}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be at most
    
    - $\min\{n, m\}$ scalar comparisons (`eq`) &
    - $|n-m|$ scalar boolean evaluations (`bool`).
    """
    return all(vecleq(v, w))

def vectrim(v, tol=None):
    r"""Remove all trailing near zero (`abs(v_i)<=tol`) coefficients.
    
    $$
        \begin{pmatrix}
            v_0 \\
            \vdots \\
            v_m
        \end{pmatrix} \ \text{where} \ m=\max\{\, j\mid |v_j|>\text{tol}\,\}\cup\{-1\} \qquad \mathbb{K}^n\to\mathbb{K}^{\leq n}
    $$
    
    `tol` may also be `None`,
    then all coefficients that evaluate to `False` are trimmed.
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar absolute evaluations (`abs`) &
    - $n$ scalar comparisons (`gt`).
    
    Notes
    -----
    - Cutting of elements that are `abs(v_i)<=tol` instead of `abs(v_i)<tol` to
    allow cutting of elements that are exactly zero by `trim(v, 0)` instead
    of `trim(v, sys.float_info.min)`.
    """
    r, t = [], []
    for x in v:
        t.append(x)
        if (x if tol is None else abs(x)>tol):
            r.extend(t)
            t.clear()
    return tuple(r)

def vecrshift(v, n, zero=0):
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
    return tuple(veclrshift(v, n, zero=zero))

def veclshift(v, n):
    r"""Shift coefficients down.
    
    $$
        (v_{i+n})_i \qquad \begin{pmatrix}
            v_n \\
            v_{n+1} \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{\max\{m-n, 0\}}
    $$
    """
    return tuple(vecllshift(v, n))
