from ..lazy import veclpos, veclneg, vecladd, vecladdc, veclsub, veclsubc, veclmul, veclrmul, vecltruediv, veclfloordiv, veclmod
from itertools import islice, repeat
from operationcounter import group_ordinal, sum_default
from typing import Any
from collections.abc import Iterable, MutableSequence



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



def vecpos(v:Iterable[Any]) -> tuple[Any,...]:
    r"""Return the identity.
    
    $$
        +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar unary plus operations (`pos`).
    """
    return tuple(veclpos(v))

def vecipos(v:MutableSequence[Any]) -> MutableSequence[Any,...]:
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


def vecneg(v:Iterable[Any]) -> tuple[Any,...]:
    r"""Return the negation.
    
    $$
        -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar negations (`neg`).
    """
    return tuple(veclneg(v))

def vecineg(v:MutableSequence[Any]) -> MutableSequence[Any,...]:
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


def vecadd(*vs:Iterable[Any]) -> tuple[Any,...]:
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
    return tuple(vecladd(*vs))

def veciadd(v:MutableSequence[Any], *ws:Iterable[Any]) -> MutableSequence[Any]:
    r"""Add.
    
    $$
        \vec{v} += \vec{w}_0+\vec{w}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`veciaddc`][vector.inplace.vectorspace.veciaddc]
    """
    it = map(sum_default, group_ordinal(*ws))
    for i, wi in enumerate(islice(it, len(v))):
        v[i] += wi
    v.extend(+wi for wi in it)
    return v


def vecaddc(v:Iterable[Any], c:Any, i:int=0, zero:Any=0) -> tuple[Any,...]:
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
    return tuple(vecladdc(v, c, i=i, zero=zero))

def veciaddc(v:MutableSequence[Any], c:Any, i:int=0, zero:Any=0) -> MutableSequence[Any]:
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


def vecsub(v:Iterable[Any], w:Iterable[Any]) -> tuple[Any,...]:
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
    return tuple(veclsub(v, w))

def vecisub(v:MutableSequence[Any], w:Iterable[Any]) -> MutableSequence[Any]:
    r"""Subtract.
    
    $$
        \vec{v} -= \vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`vecisubc`][vector.inplace.vectorspace.vecisubc]
    """
    it = iter(w)
    for i, wi in enumerate(islice(it, len(v))):
        v[i] -= wi
    v.extend(-wi for wi in it)
    return v


def vecsubc(v:Iterable[Any], c:Any, i:int=0, zero:Any=0) -> tuple[Any,...]:
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
    return tuple(veclsubc(v, c, i=i, zero=zero))

def vecisubc(v:MutableSequence[Any], c:Any, i:int=0, zero:Any=0) -> MutableSequence[Any]:
    r"""Subtract a basis vector.
    
    $$
        \vec{v} -= c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i+1\}}
    $$
    
    More efficient than `vecisub(v, vecibasis(i, c))`.
    
    See also
    --------
    - for difference on more coefficients: [`vecisub`][vector.inplace.vectorspace.vecisub]
    """
    if i < len(v):
        v[i] -= c
    else:
        v.extend(repeat(zero, i-len(v)))
        v.append(-c)
    return v


def vecmul(v:Iterable[Any], a:Any) -> tuple[Any,...]:
    r"""Return the product.
    
    $$
        \vec{v}a \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar multiplications (`rmul`).
    """
    return tuple(veclmul(v, a))

def vecrmul(a:Any, v:Iterable[Any]) -> tuple[Any,...]:
    r"""Return the product.
    
    $$
        a\vec{v} \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar multiplications (`rmul`).
    """
    return tuple(veclrmul(a, v))

def vecimul(v:MutableSequence[Any], a:Any) -> MutableSequence[Any]:
    r"""Multiply.
    
    $$
        \vec{v} \cdot= a \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    for i in range(len(v)):
        v[i] *= a
    return v


def vectruediv(v:Iterable[Any], a:Any) -> tuple[Any,...]:
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
    return tuple(vecltruediv(v, a))

def vecitruediv(v:MutableSequence[Any], a:Any) -> MutableSequence[Any]:
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


def vecfloordiv(v:Iterable[Any], a:Any) -> tuple[Any,...]:
    r"""Return the floor quotient.
    
    $$
        \left\lfloor\frac{\vec{v}}{a}\right\rfloor \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar floor divisions (`floordiv`).
    """
    return tuple(veclfloordiv(v, a))

def vecifloordiv(v:MutableSequence[Any], a:Any) -> MutableSequence[Any]:
    r"""Floor divide.
    
    $$
        \vec{v} //= a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    for i in range(len(v)):
        v[i] //= a
    return v


def vecmod(v:Iterable[Any], a:Any) -> tuple[Any,...]:
    r"""Return the remainder.
    
    $$
        \vec{v} \bmod a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar modulos (`mod`).
    """
    return tuple(veclmod(v, a))

def vecimod(v:MutableSequence[Any], a:Any) -> MutableSequence[Any]:
    r"""Mod.
    
    $$
        \vec{v} \%= a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    for i in range(len(v)):
        v[i] %= a
    return v


def vecdivmod(v:Iterable[Any], a:Any) -> tuple[tuple[Any,...], tuple[Any,...]]:
    r"""Return the floor quotient and remainder.
    
    $$
        \left\lfloor\frac{\vec{v}}{a}\right\rfloor, \ \left(\vec{v} \bmod a\right) \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    
    Complexity
    ----------
    For a vector of length $n$ there will be
    
    - $n$ scalar divmods (`divmod`).
    """
    q, r = [], []
    for vi in v:
        qi, ri = divmod(vi, a)
        q.append(qi)
        r.append(ri)
    return tuple(q), tuple(r)
