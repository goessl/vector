# vector

An infinite-dimensional vector package.
```python
>>> from vector import vecadd
>>> vecadd((1, 2), (4, 5, 6))
(5, 7, 6)
>>> 
>>> from vector import Vector
>>> v = Vector((1, 2))
>>> w = Vector((4, 5, 6))
>>> v + w
Vector(5, 7, 6, ...)
>>> 
>>> from vector import vecnpadd
>>> vecnpadd((1, 2), ((3, 4, 5),
...                   (6, 7, 8)))
array([[4, 6, 5],
       [7, 9, 8]])
```

## Installation

```console
pip install git+https://github.com/goessl/vector.git
```

## Usage

This package includes
- general-purpose *functions* (prefixed `vec`),
- a clean *class* (`Vector`)&
- improved *numpy-routines* (prefixed `vecnp`)

to handle infinite-dimensional vectors.
It operates on vectors of different lengths, treating them as infinite-dimensional by assuming that all components after the given ones are *zero*.

### Functions

```python
>>> from vector import vecadd
>>> a = (5, 6, 7)
>>> b = [2]
>>> c = range(4)
>>> vecadd(a, b, c)
(7, 7, 9, 3)
```

All functions *accept single exhaustible iterables*.

They *return vectors as tuples*.

The functions are *type-independent*. However, the data types used must *support necessary scalar operations*. For instance, for vector addition, components must be addable — this may include operations with padded integer zeros. Empty operations return the zero vector (e.g. `vecadd()==veczero`) or integer zeros (e.g. `vecdot(veczero, veczero)==int(0)`).

Padding is done with `int(0)`.

creation stuff
- `veczero = ()`: Zero vector.
- `vecbasis(i, c=1)`: Return the `i`-th basis vector times `c`. The returned value is a tuple with `i` integer zeros followed by `c`.
- `vecrand(n)`: Return a random vector of `n` uniform coefficients in `[0, 1[`.
- `vecrandn(n, normed=True, mu=0, sigma=1)`: Return a random vector of `n` normal distributed coefficients.

utility stuff
- `veceq(v, w)`: Return if two vectors are equal.
- `vectrim(v, tol=1e-9)`: Remove all trailing near zero (<=tol) coefficients.
- `vecround(v, ndigits=None)`: Round all coefficients to the given precision.

Hilbert space stuff
- `vecabsq(v)`: Return the sum of absolute squares of the coefficients.
- `vecabs(v)`: Return the Euclidean/L2-norm.
- `vecdot(v, w)`: Return the inner product of two vectors without conjugation.
- `vecparallel(v, w)`: Return if two vectors are parallel.

vector space stuff
- `vecpos(v)`: Return the vector with the unary positive operator applied.
- `vecneg(v)`: Return the vector with the unary negative operator applied.
- `vecadd(*vs)`: Return the sum of vectors.
- `vecsub(v, w)`: Return the difference of two vectors.
- `vecmul(a, v)`: Return the product of a scalar and a vector.
- `vectruediv(v, a)`: Return the true division of a vector and a scalar.
- `vecfloordiv(v, a)`: Return the floor division of a vector and a scalar.

elementwise stuff
- `vechadamard(*vs)`: Return the elementwise product of vectors.
- `vechadamardtruediv(v, w)`: Return the elementwise true division of two vectors.
- `vechadamardfloordiv(v, w)`: Return the elementwise floor division of two vectors.
- `vechadamardmod(v, w)`: Return the elementwise mod of two vectors.
- `vechadamardmin(*vs)`: Return the elementwise minimum of vectors.
- `vechadamardmax(*vs)`: Return the elementwise maximum of vectors.

### Class

The immutable `Vector` class wraps all the mentioned functions into a tidy package, making them easier to use by enabling interaction through operators.

Its coefficients are internally stored as a tuple in the `coef` attribute and therefore *zero-indexed*.

Vector operations return the same type (`type(v+w)==type(v)`) so the class can easily be extended (to e.g. a polynomial class).

initialisation stuff
- `Vector(i)`: Create a new vector with the given coefficients or the `i`-th basis vector if an integer `i` is given.
- `Vector.rand(n)`: Create a random vector of `n` uniform coefficients in `[0, 1[`.
- `Vector.randn(n, normed=True, mu=0, sigma=1))`: Create a random vector of `n` normal distributed coefficients.
- `Vector.ZERO`: Zero vector.

```python
>>> from vector import Vector
>>> Vector((1, 2, 3))
Vector(1, 2, 3, ...)
>>> Vector.randn(3)
Vector(-0.5613820142699765, -0.028308921297709365, 0.8270724508948077, ...)
>>> Vector(3)
Vector(0, 0, 0, 1, ...)
```

sequence stuff
- `len(v)`: Return the number of set coefficients.
- `v[key]`: Return the indexed coefficient or coefficients. Not set coefficients default to 0.
- `iter(v)`: Return an iterator over the set coefficients.
- `v == w`: Return if of same type with same coefficients.
- `v << i`: Return a vector with coefficients shifted to lower indices.
- `v >> i`: Return a vector with coefficients shifted to higher indices.

utility stuff
- `v.trim(tol=1e-9)`: Remove all trailing near zero (abs<=tol) coefficients.
- `v.round(ndigits=None)`: Round all coefficients to the given precision.

Hilbert space stuff
- `v.absq()`: Return the sum of absolute squares of the coefficients.
- `abs(v)`: Return the Euclidean/L2-norm. Return the square root of `vecabsq`.
- `v @ w`: Return the inner product of two vectors without conjugation.

vector space stuff
- `v + w`: Return the vector sum.
- `v - w`: Return the vector difference.
- `v * a`: Return the scalar product.
- `v / a`: Return the scalar true division.
- `v // a`: Return the scalar floor division.
- `v % a`: Return the elementwise mod with a scalar.

elementwise stuff
- `v.hadamard(w)`: Return the elementwise product with another vector.
- `v.hadamardtruediv(w)`: Return the elementwise true division with another vector.
- `v.hadamardfloordiv(w)`: Return the elementwise floor division with another vector.
- `v.hadamardmod(w)`: Return the elementwise mod with another vector.

### `numpy`-routines

```python
>>> from vector import vecnpadd
>>> vecnpadd((1, 2), ((3, 4, 5),
...                   (6, 7, 8)))
array([[4, 6, 5],
       [7, 9, 8]])
```

`numpy`-versions of the functions are also provided, to *operate on multiple vectors* at once. They behave like the ones in `numpy.polynomial.polynomial`, but *also work on 2D-arrays* (and *all combinations of 1D & 2D arrays*) and broadcast to multiple dimensions like the usual `numpy` operations (but adjust the shapes accordingly).

*`vecnpzero` is `np.array([0])`* like `numpy.polynomial.polynomial.polyzero`, not `veczero=()` (empty tuple, no zero coefficient left) like in the functions and class above.

Padding is done with `numpy.int64(0)`.

They return scalars or `numpy.ndarray`s.

Creation routines have a dimension argument `d`. If left to `None`, the returned values are 1D, so a single vector. If given, the routines return a 2D-array representing multiple vectors in rows.

creation stuff
- `vecnpzero(d=None)`: Return `d` zero vectors. The returned value is a `(d, 1)`-array of zeros if `d` is not `None` or `[0]` otherwise.
- `vecnpbasis(i, c=1, d=None)`: Return `d` many `i`-th basis vectors times `c`. The returned value is a `(d, i+1)`-array if `d` is not `None` or `(i+1,)` otherwise.
- `vecnprand(n, d=None)`: Return `d` random vectors of `n` uniform coefficients in `[0, 1[`. The returned value is a `(d, n)`-array if `d` is not `None` or `(n,)` otherwise.
- `vecnprandn(n, normed=True, d=None)`: Return `d` random vectors of `n` normal distributed coefficients. The returned value is a `(d, n)`-array if `d` is not `None` or `(n,)` otherwise.

utility stuff
- `vecnpeq(v, w)`: Return if two vectors are equal.
- `vecnptrim(v, tol=1e-9)`: Remove all trailing near zero (abs(v_i)<=tol) coefficients.
- (`numpy.round` already exists)

Hilbert space stuff
- `vecnpabsq(v)`: Return the sum of absolute squares of the coefficients.
- `vecnpabs(v)`: Return the Euclidean/L2-norm.
- `vecnpdot(v, w)`: Return the inner product of two vectors without conjugation.
- `vecnpparallel(v, w)`: Return if two vectors are parallel.

vector space stuff
- `vecnpadd(*vs)`: Return the sum of vectors.
- `vecnpsub(v, w)`: Return the difference of two vectors.

## Design

### Prefix

No prefix? Could use no prefix to be more pure, like `add` instead of `vecadd`, but then you would always have to use `from vec import add as vecadd` if used with other libraries (like `operator`).

Also avoids keyword collisions (`abs` is reserved, `vecabs` isn't).

Do it like `numpy.polynomial.polynomial. ...`.

### `truediv`

Why called `truediv` instead of `div`.
`div` would be more appropriate for an absolute clean mathematical implementation, that doesn't care about the language used.
But the package might be used for pure integers/integer arithmetic.
`truediv`/`floordiv` is unambiguous.

Like Python `operator`s.

### `vecabsq(v)`

Reasons why it exists:
- Occours in math.
- Most importantly: type independent because it doesn't use `sqrt`.

### `trim`

cutting of elements that are `abs(vi)<=tol` instead of `abs(vi)<tol` to allow cutting of exactly just zeros by `trim(v, 0)` instead of `trim(v, sys.float_info.min)`.

`tol=1e-9` like in https://peps.python.org/pep-0485/#defaults

### `rand & randn`

Naming like in `numpy` because seems more concise (not `random` & `gauss` as in the stdlib).

### `Vector.__init__()`

By iterable or integer for basis vector?
- Provide signature like `min` (single argument=iterable or multiple args)? No, because this way a single integer can't be distinguished to mean a single coefficient or a basis vector.
- Automatically trim on creation? Nah, do nothing without specially being told to do so.

## todo

 - [ ] `zip` version between `zip` & `zip_longest`. Yields different sized tuples.
 - [x] docstrings
 - [x] `numpy` routines

## License (MIT)

Copyright (c) 2022-2025 Sebastian Gössl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
