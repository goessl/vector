# Design

## Prefix

No prefix? Could use no prefix to be more pure, like `add` instead of `vecadd`, but then you would always have to use `from vec import add as vecadd` if used with other libraries (like `operator`).

Also avoids keyword collisions (`abs` is reserved, `vecabs` isn't).

Do it like `numpy.polynomial.polynomial. ...`.

## `truediv`

Why called `truediv` instead of `div`.
`div` would be more appropriate for an absolute clean mathematical implementation, that doesn't care about the language used.
But the package might be used for pure integers/integer arithmetic.
`truediv`/`floordiv` is unambiguous.

Like Python `operator`s.

## `vecabsq(v)`

Reasons why it exists:

- Occurs in math.
- Most importantly: type independent because it doesn't use `sqrt`.

## `trim`

cutting of elements that are `abs(vi)<=tol` instead of `abs(vi)<tol` to allow cutting of exactly just zeros by `trim(v, 0)` instead of `trim(v, sys.float_info.min)`.

`tol=1e-9` like in [PEP 485](https://peps.python.org/pep-0485/#defaults).

## `rand & randn`

Naming like in `numpy` because seems more concise (not `random` & `gauss` as in the stdlib).

## `Vector.__init__()`

By iterable or integer for basis vector?

- Provide signature like `min` (single argument=iterable or multiple args)? No, because this way a single integer can't be distinguished to mean a single coefficient or a basis vector.

- Automatically trim on creation? Nah, do nothing without specially being told to do so.
