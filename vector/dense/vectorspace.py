from ..lazy import veclpos, veclneg, vecladd, vecladdc, veclsub, veclsubc, veclmul, veclrmul, vecltruediv, veclfloordiv, veclmod
from itertools import islice, repeat
from operationcounter import group_ordinal, sum_default
from typing import Any, TypeVar
from collections.abc import Callable, Iterable, MutableSequence, Sequence



__all__ = ('vecpos',                 'vecipos',
           'vecneg',                 'vecineg',
           'vecadd',                 'veciadd',
           'vecaddc',                'veciaddc',
           'vecsub',                 'vecisub',
           'vecsubc',                'vecisubc',
           'vecmul',      'vecrmul', 'vecimul',
           'vectruediv',             'vecitruediv',
           'vecfloordiv',            'vecifloordiv',
           'vecmod',                 'vecimod',
           'vecdivmod')



S = TypeVar('S', bound=Sequence)
M = TypeVar('M', bound=MutableSequence)



def vecpos(v:Iterable, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the identity.
    
    $$
        +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar unary plus operations (`pos`).
    """
    factory = type(v) if factory is None else factory
    return factory(veclpos(v))

def vecipos(v:M) -> M[Any,...]:
    r"""Apply the unary plus operator.
    
    $$
        +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar unary plus operations (`pos`).
    """
    for i, vi in enumerate(v):
        v[i] = +vi
    return v


def vecneg(v:Iterable, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the negation.
    
    $$
        -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar negations (`neg`).
    """
    factory = type(v) if factory is None else factory
    return factory(veclneg(v))

def vecineg(v:M) -> M[Any,...]:
    r"""Negate.
    
    $$
        -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar unary negations (`neg`).
    """
    for i, vi in enumerate(v):
        v[i] = -vi
    return v


def vecadd(*vs:Iterable, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the sum.
    
    $$
        \vec{v}_0+\vec{v}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ scalar additions (`add`).
    
    See also
    --------
    - for sum on a single coefficient: [`vecaddc`][vector.dense.vectorspace.vecaddc]
    """
    if factory is None:
        factory = type(vs[0]) if vs else tuple
    return factory(vecladd(*vs))

def veciadd(v:M, *ws:Iterable) -> M:
    r"""Add.
    
    $$
        \vec{v} += \vec{w}_0+\vec{w}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`veciaddc`][vector.dense.vectorspace.veciaddc]
    """
    it = map(sum_default, group_ordinal(*ws))
    for i, wi in enumerate(islice(it, len(v))):
        v[i] += wi
    v.extend(+wi for wi in it)
    return v


def vecaddc(v:Iterable, c:Any, i:int=0, zero:Any=0, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the sum with a basis vector.
    
    $$
        \vec{v}+c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i+1\}}
    $$
    
    More efficient than `vecadd(v, vecbasis(i, c))`.
    
    Complexity
    ----------
    There will be
    
    - one scalar addition (`add`) if $i\le n$ or
    - one unary plus operations (`pos`) otherwise.
    
    See also
    --------
    - for sum on more coefficients: [`vecadd`][vector.dense.vectorspace.vecadd]
    """
    factory = type(v) if factory is None else factory
    return factory(vecladdc(v, c, i=i, zero=zero))

def veciaddc(v:M, c:Any, i:int=0, zero:Any=0) -> M:
    r"""Add a basis vector.
    
    $$
        \vec{v} += c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i+1\}}
    $$
    
    More efficient than `veciaddc(v, vecibasis(i, c))`.
    
    See also
    --------
    - for sum on more coefficients: [`veciadd`][vector.dense.vectorspace.veciadd]
    """
    if i < len(v):
        v[i] += c
    else:
        v.extend(repeat(zero, i-len(v)))
        v.append(+c)
    return v


def vecsub(v:Iterable, w:Iterable, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the difference.
    
    $$
        \vec{v}-\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    
    Complexity
    ----------
    For two vectors of lengths $n$ & $m$ there will be
    
    - $\min\{n, m\}$ scalar subtractions (`sub`) &
    - $\begin{cases}m-n&m\ge n\\0&m\le n\end{cases}$ negations (`neg`).
    
    See also
    --------
    - for difference on a single coefficient: [`vecsubc`][vector.dense.vectorspace.vecsubc]
    """
    factory = type(v) if factory is None else factory
    return factory(veclsub(v, w))

def vecisub(v:M, w:Iterable) -> M:
    r"""Subtract.
    
    $$
        \vec{v} -= \vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`vecisubc`][vector.dense.vectorspace.vecisubc]
    """
    it = iter(w)
    for i, wi in enumerate(islice(it, len(v))):
        v[i] -= wi
    v.extend(-wi for wi in it)
    return v


def vecsubc(v:Iterable, c:Any, i:int=0, zero:Any=0, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the difference with a basis vector.
    
    $$
        \vec{v}-c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i+1\}}
    $$
    
    More efficient than `vecsub(v, vecbasis(i, c))`.
    
    Complexity
    ----------
    There will be
    
    - one scalar subtraction (`sub`) if $i\le n$ or
    - one scalar negation (`neg`) otherwise.
    
    See also
    --------
    - for difference on more coefficients: [`vecsub`][vector.dense.vectorspace.vecsub]
    """
    factory = type(v) if factory is None else factory
    return factory(veclsubc(v, c, i=i, zero=zero))

def vecisubc(v:M, c:Any, i:int=0, zero:Any=0) -> M:
    r"""Subtract a basis vector.
    
    $$
        \vec{v} -= c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i+1\}}
    $$
    
    More efficient than `vecisub(v, vecibasis(i, c))`.
    
    See also
    --------
    - for difference on more coefficients: [`vecisub`][vector.dense.vectorspace.vecisub]
    """
    if i < len(v):
        v[i] -= c
    else:
        v.extend(repeat(zero, i-len(v)))
        v.append(-c)
    return v


def vecmul(v:Iterable, a:Any, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the product.
    
    $$
        \vec{v}a \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar multiplications (`rmul`).
    """
    factory = type(v) if factory is None else factory
    return factory(veclmul(v, a))

def vecrmul(a:Any, v:Iterable, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the product.
    
    $$
        a\vec{v} \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar multiplications (`rmul`).
    """
    factory = type(v) if factory is None else factory
    return factory(veclrmul(a, v))

def vecimul(v:M, a:Any) -> M:
    r"""Multiply.
    
    $$
        \vec{v} \cdot= a \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    for i in range(len(v)):
        v[i] *= a
    return v


def vectruediv(v:Iterable, a:Any, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the true quotient.
    
    $$
        \frac{\vec{v}}{a} \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar true divisions (`truediv`).
    
    Notes
    -----
    Why called `truediv` instead of `div`?
    
    - `div` would be more appropriate for an absolute clean mathematical
    implementation, that doesn't care about the language used. But the package
    might be used for pure integers/integer arithmetic, so both, `truediv`
    and `floordiv` operations have to be provided, and none should be
    privileged over the other by getting the universal `div` name.
    - `truediv`/`floordiv` is unambiguous, like Python `operator`s.
    """
    factory = type(v) if factory is None else factory
    return factory(vecltruediv(v, a))

def vecitruediv(v:M, a:Any) -> M:
    r"""True divide.
    
    $$
        \vec{v} /= a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Notes
    -----
    Why called `truediv` instead of `div`?
    
    - `div` would be more appropriate for an absolute clean mathematical
    implementation, that doesn't care about the language used. But the package
    might be used for pure integers/integer arithmetic, so both, `truediv`
    and `floordiv` operations have to be provided, and none should be
    privileged over the other by getting the universal `div` name.
    - `truediv`/`floordiv` is unambiguous, like Python `operator`s.
    """
    for i in range(len(v)):
        v[i] /= a
    return v


def vecfloordiv(v:Iterable, a:Any, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the floor quotient.
    
    $$
        \left\lfloor\frac{\vec{v}}{a}\right\rfloor \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar floor divisions (`floordiv`).
    """
    factory = type(v) if factory is None else factory
    return factory(veclfloordiv(v, a))

def vecifloordiv(v:M, a:Any) -> M:
    r"""Floor divide.
    
    $$
        \vec{v} //= a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    for i in range(len(v)):
        v[i] //= a
    return v


def vecmod(v:Iterable, a:Any, factory:Callable[[Iterable],S]|None=None) -> S:
    r"""Return the remainder.
    
    $$
        \vec{v} \bmod a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar modulos (`mod`).
    """
    factory = type(v) if factory is None else factory
    return factory(veclmod(v, a))

def vecimod(v:M, a:Any) -> M:
    r"""Mod.
    
    $$
        \vec{v} \%= a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    for i in range(len(v)):
        v[i] %= a
    return v


def vecdivmod(v:Iterable, a:Any, factory:Callable[[Iterable],S]|None=None) -> tuple[S, S]:
    r"""Return the floor quotient and remainder.
    
    $$
        \left\lfloor\frac{\vec{v}}{a}\right\rfloor, \ \left(\vec{v} \bmod a\right) \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar divmods (`divmod`).
    """
    factory = type(v) if factory is None else factory
    q, r = [], []
    for vi in v:
        qi, ri = divmod(vi, a)
        q.append(qi)
        r.append(ri)
    return factory(q), factory(r)
