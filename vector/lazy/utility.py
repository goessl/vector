from itertools import chain, islice, repeat, zip_longest



__all__ = ('vecleq', 'vecltrim', 'veclrshift', 'vecllshift')



def vecleq(v, w):
    r"""Return if two vectors are equal.
    
    $$
        \vec{v}=\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{B}
    $$
    """
    sentinel = object()
    for vi, wi in zip_longest(v, w, fillvalue=sentinel):
        if wi is sentinel:
            yield not bool(vi)
        elif vi is sentinel:
            yield not bool(wi)
        else:
            yield vi == wi

def vecltrim(v, tol=1e-9):
    r"""Remove all trailing near zero (`abs(v_i)<=tol`) coefficients.
    
    $$
        \begin{pmatrix}
            v_0 \\
            \vdots \\
            v_m
        \end{pmatrix} \ \text{where} \ m=\max\{\, j\mid |v_{j-1}|>\text{tol}\,\}\cup\{-1\} \qquad \mathbb{K}^n\to\mathbb{K}^{\leq n}
    $$
    
    `tol` may also be `None`,
    then all coefficients that evaluate to `False` are trimmed.
    
    Notes
    -----
    - Cutting of elements that are `abs(v_i)<=tol` instead of `abs(v_i)<tol` to
    allow cutting of elements that are exactly zero by `trim(v, 0)` instead
    of `trim(v, sys.float_info.min)`.
    - `tol=1e-9` like in [PEP 485](https://peps.python.org/pep-0485/#defaults).
    """
    t = []
    for x in v:
        t.append(x)
        if (x if tol is None else abs(x)>tol):
            yield from t
            t.clear()

def veclrshift(v, n, zero=0):
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
    yield from chain(repeat(zero, n), v)

def vecllshift(v, n):
    r"""Shift coefficients down.
    
    $$
        (v_{i+n})_i \qquad \begin{pmatrix}
            v_n \\
            v_{n+1} \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{\max\{m-n, 0\}}
    $$
    """
    yield from islice(v, n, None)
