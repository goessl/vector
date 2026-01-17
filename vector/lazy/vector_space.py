from operator import pos, neg, mul, truediv, floordiv, mod
from itertools import chain, islice, repeat, zip_longest
from functools import partial
from operationcounter import MISSING, group_ordinal, sum_default



__all__ = ('veclpos', 'veclneg', 'vecladd', 'vecladdc', 'veclsub', 'veclsubc',
           'veclmul', 'veclrmul', 'vecltruediv', 'veclfloordiv', 'veclmod', 'vecldivmod')



def veclpos(v):
    r"""Return the identity.
    
    $$
        +\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    yield from map(pos, v)

def veclneg(v):
    r"""Return the negation.
    
    $$
        -\vec{v} \qquad \mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    yield from map(neg, v)

def vecladd(*vs):
    r"""Return the sum.
    
    $$
        \vec{v}_0+\vec{v}_1+\cdots \qquad \mathbb{K}^{n_0}\times\mathbb{K}^{n_1}\times\cdots\to\mathbb{K}^{\max_i n_i}
    $$
    
    See also
    --------
    - for sum on a single coefficient: [`vecladdc`][vector.lazy.vector_space.vecladdc]
    """
    yield from map(partial(sum_default, default=MISSING), group_ordinal(*vs))

def vecladdc(v, c, i=0, zero=0):
    r"""Return the sum with a basis vector.
    
    $$
        \vec{v}+c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i+1\}}
    $$
    
    More efficient than `vecladd(v, veclbasis(i, c))`.
    
    See also
    --------
    - for sum on more coefficients: [`vecladd`][vector.lazy.vector_space.vecladd]
    """
    v = iter(v)
    yield from islice(chain(v, repeat(zero)), i)
    try:
        yield next(v) + c
    except StopIteration:
        yield +c
    yield from v

def veclsub(v, w):
    r"""Return the difference.
    
    $$
        \vec{v}-\vec{w} \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max\{m, n\}}
    $$
    
    See also
    --------
    - for difference on a single coefficient: [`veclsubc`][vector.lazy.vector_space.veclsubc]
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
    r"""Return the difference with a basis vector.
    
    $$
        \vec{v}-c\vec{e}_i \qquad \mathbb{K}^n\to\mathbb{K}^{\max\{n, i+1\}}
    $$
    
    More efficient than `veclsub(v, veclbasis(i, c))`.
    
    See also
    --------
    - for difference on more coefficients: [`veclsub`][vector.lazy.vector_space.veclsub]
    """
    v = iter(v)
    yield from islice(chain(v, repeat(zero)), i)
    try:
        yield next(v) - c
    except StopIteration:
        yield -c
    yield from v

def veclmul(v, a):
    r"""Return the product.
    
    $$
        \vec{v}a \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    yield from map(mul, v, repeat(a))

def veclrmul(a, v):
    r"""Return the product.
    
    $$
        a\vec{v} \qquad \mathbb{K}\times\mathbb{K}^n\to\mathbb{K}^n
    $$
    """
    yield from map(mul, repeat(a), v)

def vecltruediv(v, a):
    r"""Return the true quotient.
    
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
    r"""Return the floor quotient.
    
    $$
        \left\lfloor\frac{\vec{v}}{a}\right\rfloor \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    yield from map(floordiv, v, repeat(a))

def veclmod(v, a):
    r"""Return the remainder.
    
    $$
        \vec{v} \bmod a \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n
    $$
    """
    yield from map(mod, v, repeat(a))

def vecldivmod(v, a):
    r"""Return the floor quotient and remainder.
    
    $$
        \left\lfloor\frac{\vec{v}}{a}\right\rfloor, \ \left(\vec{v} \bmod a\right) \qquad \mathbb{K}^n\times\mathbb{K}\to\mathbb{K}^n\times\mathbb{K}^n
    $$
    """
    for vi in v:
        yield divmod(vi, a)
