from operator import pos, neg, mul, truediv, floordiv, mod
from itertools import chain, islice, repeat, zip_longest
from functools import partial
from operationcounter import MISSING, group_ordinal, sum_default



__all__ = ('veclpos', 'veclneg', 'vecladd', 'vecladdc', 'veclsub', 'veclsubc',
           'veclmul', 'vecltruediv', 'veclfloordiv', 'veclmod', 'vecldivmod')



def veclpos(v):
    r"""Return the vector with the unary positive operator applied.
    
    $$
        +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    yield from map(pos, v)

def veclneg(v):
    r"""Return the vector with the unary negative operator applied.
    
    $$
        -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    yield from map(neg, v)

def vecladd(*vs):
    r"""Return the sum of vectors.
    
    $$
        \vec{v}_0+\vec{v}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    """
    yield from map(partial(sum_default, default=MISSING), group_ordinal(*vs))

def vecladdc(v, c, i=0, zero=0):
    r"""Return `v` with `c` added to the `i`-th coefficient.
    
    $$
        \vec{v}+c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i\}}
    $$
    
    More efficient than `vecladd(v, veclbasis(i, c))`.
    """
    v = iter(v)
    yield from islice(chain(v, repeat(zero)), i)
    try:
        yield next(v) + c
    except StopIteration:
        yield +c
    yield from v

def veclsub(v, w):
    r"""Return the difference of two vectors.
    
    $$
        \vec{v}-\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    """
    sentinel = object()
    for vi, wi in zip_longest(v, w, fillvalue=sentinel):
        if wi is sentinel:
            yield vi
        elif vi is sentinel:
            yield -wi
        else:
            yield vi - wi

def veclsubc(v, c, i=0, zero=0):
    r"""Return `v` with `c` subtracted from the `i`-th coefficient.
    
    $$
        \vec{v}-c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i\}}
    $$
    
    More efficient than `veclsub(v, veclbasis(i, c))`.
    """
    v = iter(v)
    yield from islice(chain(v, repeat(zero)), i)
    try:
        yield next(v) - c
    except StopIteration:
        yield -c
    yield from v

def veclmul(a, v):
    r"""Return the product of a scalar and a vector.
    
    $$
        a\vec{v} \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    yield from map(mul, repeat(a), v)

def vecltruediv(v, a):
    r"""Return the true division of a vector and a scalar.
    
    $$
        \frac{\vec{v}}{a} \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
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
    yield from map(truediv, v, repeat(a))

def veclfloordiv(v, a):
    r"""Return the floor division of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    yield from map(floordiv, v, repeat(a))

def veclmod(v, a):
    r"""Return the elementwise mod of a vector and a scalar.
    
    $$
        \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    yield from map(mod, v, repeat(a))

def vecldivmod(v, a):
    r"""Return the elementwise divmod of a vector and a scalar.
    
    $$
        \left(\left\lfloor\frac{v_i}{a}\right\rfloor\right)_i, \ \left(v_i \mod a\right)_i \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    """
    for vi in v:
        yield divmod(vi, a)
