from ..lazy import vecleq, veclround, veclrshift, vecllshift



__all__ = ('veclen', 'veceq', 'vectrim', 'vecround', 'vecrshift', 'veclshift')



def veclen(v):
    """Return the length (number of set coefficients) of the vector.
    
    Also works for single exhaustible iterables where `len(v)` wouldn't work,
    altough the vector is gone then.
    """
    return sum(1 for _ in v)

def veceq(v, w):
    r"""Return if two vectors are equal.
    
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

def vectrim(v, tol=1e-9):
    r"""Remove all trailing near zero (`abs(v_i)<=tol`) coefficients.
    
    $$
        \begin{pmatrix}
            v_0 \\
            \vdots \\
            v_m
        \end{pmatrix} \ \text{where} \ m=\max\{\, j\mid |v_j|>\text{tol}\,\}\cup\{-1\} \qquad \mathbb{K}^n\to\mathbb{K}^{\leq n}
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar absolute evaluations (`abs`) &
    - $n$ scalar comparisons (`gt`).
    
    Notes
    -----
    - Cutting of elements that are `abs(vi)<=tol` instead of `abs(vi)<tol` to
    allow cutting of elements that are exactly zero by `trim(v, 0)` instead
    of `trim(v, sys.float_info.min)`.
    - `tol=1e-9` like in [PEP 485](https://peps.python.org/pep-0485/#defaults).
    """
    r, t = [], []
    for x in v:
        t.append(x)
        if abs(x)>tol:
            r.extend(t)
            t.clear()
    return tuple(r)

def vecround(v, ndigits=None):
    r"""Round all coefficients to the given precision.
    
    $$
        (\text{round}_\text{ndigits}(v_i))_i \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar roundings (`round`).
    """
    return tuple(veclround(v, ndigits=ndigits))

def vecrshift(v, n, zero=0):
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
    return tuple(veclrshift(v, n, zero=zero))

def veclshift(v, n):
    r"""Remove `n` many coefficients at the beginning of the vector.
    
    $$
        (v_{i+n})_i \qquad \begin{pmatrix}
            v_n \\
            v_{n+1} \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{\max\{m-n, 0\}}
    $$
    """
    return tuple(vecllshift(v, n))
