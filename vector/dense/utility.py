from typing import Any, TypeVar
from collections.abc import Callable, Iterable, Iterator, MutableSequence
from ..lazy import vecleq, vecltrim, veclrshift, vecllshift



__all__ = ('veclen',
           'veceq',
           'vectrim',   'vecitrim',
           'vecrshift', 'vecirshift',
           'veclshift', 'vecilshift')



V = TypeVar('V')
M = TypeVar('M', bound=MutableSequence)



def veclen(v:Iterable) -> int:
    """Return the length (number of set coefficients).
    
    Doesn't handle trailing zeros, use [`vectrim`][vector.dense.utility.vectrim]
    if needed.
    
    Notes
    -----
    For generators as they have no `len`gth, altough the vector is gone then.
    """
    if hasattr(v, '__len__'):
        return len(v)
    return sum(1 for _ in v)


def veceq(v:Iterable, w:Iterable, factory:Callable[[Iterable],V]=all) -> V:
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
    factory = factory or (iter if isinstance(v, Iterator) else type(v))
    return all(vecleq(v, w))


def vectrim(v:Iterable, tol:Any|None=None, factory:Callable[[Iterable],V]|None=None) -> V:
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
    
    - $n$ scalar boolean evaluations (`bool`).
    
    Notes
    -----
    - Cutting of elements that are `abs(v_i)<=tol` instead of `abs(v_i)<tol` to
    allow cutting of elements that are exactly zero by `trim(v, 0)` instead
    of `trim(v, sys.float_info.min)`.
    """
    factory = factory or (iter if isinstance(v, Iterator) else type(v))
    return factory(vecltrim(v))

def vecitrim(v:M, tol:Any|None=None) -> M:
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
    
    Notes
    -----
    - Cutting of elements that are `abs(v_i)<=tol` instead of `abs(v_i)<tol` to
    allow cutting of elements that are exactly zero by `trim(v, 0)` instead
    of `trim(v, sys.float_info.min)`.
    """
    idx = next((i for i in reversed(range(len(v))) if v[i]), -1)
    del v[idx+1:]
    return v


def vecrshift(v:Iterable, n:int, zero:Any=0, factory:Callable[[Iterable],V]|None=None) -> V:
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
    factory = factory or (iter if isinstance(v, Iterator) else type(v))
    return factory(veclrshift(v, n, zero=zero))

def vecirshift(v:M, n:int, zero:Any=0) -> M:
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
    v[:0] = [zero] * n
    return v


def veclshift(v:Iterable, n:int, factory:Callable[[Iterable],V]|None=None) -> V:
    r"""Shift coefficients down.
    
    $$
        (v_{i+n})_i \qquad \begin{pmatrix}
            v_n \\
            v_{n+1} \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{\max\{m-n, 0\}}
    $$
    """
    factory = factory or (iter if isinstance(v, Iterator) else type(v))
    return factory(vecllshift(v, n))

def vecilshift(v:M, n:int) -> M:
    r"""Shift coefficients down.
    
    $$
        (v_{i+n})_i \qquad \begin{pmatrix}
            v_n \\
            v_{n+1} \\
            \vdots
        \end{pmatrix} \qquad \mathbb{K}^m\to\mathbb{K}^{\max\{m-n, 0\}}
    $$
    """
    del v[:n]
    return v
