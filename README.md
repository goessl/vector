# vector

An infinite-dimensional vector package.
```python
>>> from vector import vecadd
>>> vecadd((1, 2), (4, 5, 6))
(5, 7, 6)
>>> from vector import Vector
>>> v = Vector((1, 2))
>>> w = Vector((4, 5, 6))
>>> v+w
Vector(5, 7, 6, ...)
```

## Installation

```console
pip install git+https://github.com/goessl/vector.git
```

## Usage

This package offers a suite of minimalistic, general-purpose *functions* alongside a clean, higher-level, operator overloaded *class* designed to handle infinite-dimensional vectors.
It operates on vectors of different lengths, treating them as infinite-dimensional by assuming that all components after the given ones are *integer zeros*.

### Functions

```python
>>> from vector import vecadd
>>> a = (5, 6, 7)
>>> b = [2]
>>> c = range(4)
>>> vecadd(a, b, c)
(7, 7, 9, 3)
```

All functions *accept sequences*, most even *single exhaustible iterables*.

They *return vectors as tuples*.

The functions are *type-independent*. However, the data types used must *support necessary scalar operations*. For instance, for vector addition, components must be addable — this may include operations with padded integer zeros. Empty sums or dot products of empty vectors return integer zeros.

creation stuff
- `veczero = ()`: Zero vector.
- `vecbasis(i, c=1)`: Return the `i`-th basis vector times `c`. The retured value is a tuple with `i` integer zeros followed by `c`.
- `def vecrandom(n)`: Return a random vector of `n` uniform coefficients in `[0, 1[`.
- `def vecgauss(n, normed=True, mu=0, sigma=1)`: Return a random vector of `n` normal distributed coefficients.

sequence stuff
- `veceq(v, w)`: Return if two vectors are equal.
- `vectrim(v, tol=1e-9)`: Remove all trailing near zero (<=tol) coefficients.
- `vecround(v, ndigits=None)`: Round all coefficients to the given precision.

Hilbert space stuff
- `vecabsq(v)`: Return the sum of absolute squares of the coefficients.
- `vecabs(v)`: Return the Euclidean/L2-norm.
- `vecdot(v, w)`: Return the real dot product of two vectors. No argument is complex conjugated. All coefficients are used as is.

vector space stuff
- `vecadd(*vs)`: Return the sum of vectors.
- `vecsub(v, w)`: Return the difference of two vectors.
- `vecmul(a, v)`: Return the product of a scalar and a vector.
- `vectruediv(v, a)`: Return the true division of a vector and a scalar.
- `vecfloordiv(v, a)`: Return the floor division of a vector and a scalar.

### Class

The immutable `Vector` class wraps all the mentioned functions into a tidy package, making them easier to use by enabling interaction through operators.
Its coefficients are internally stored as a tuple in the `coef` attribute and therefore *zero-indexed*.

initialisation stuff
- `Vector(i)`: Create a new vector with the given coefficients or the `i`-th basis vector if an integer `i` is given.
- `Vector.random(n)`: Create a random vector of `n` uniform coefficients in `[0, 1[`.
- `Vector.gauss(n, normed=True, mu=0, sigma=1))`: Create a random vector of `n` normal distributed coefficients.
- `Vector.ZERO`: Zero vector.

```python
>>> from vector import Vector
>>> Vector((1, 2, 3))
Vector(1, 2, 3, ...)
>>> Vector.gauss(3)
Vector(-0.5613820142699765, -0.028308921297709365, 0.8270724508948077, ...)
>>> Vector(3)
Vector(0, 0, 0, 1, ...)
```

container and sequence stuff
- `len(v)`: Return the number of set coefficients.
- `v[key]`: Return the indexed coefficient or coefficients. Not set coefficients default to 0.
- `iter(v)`: Return an iterator over the set coefficients.
- `v == w`: Return if of same type with same coefficients.
- `v << i`: Return a vector with coefficients shifted to lower indices.
- `v >> i`: Return a vector with coefficients shifted to higher indices.
- `v.trim(tol=1e-9)`: Remove all trailing near zero (abs<=tol) coefficients.
- `v.round(ndigits=None)`: Round all coefficients to the given precision.

Hilbert space stuff
- `v.absq()`: Return the sum of absolute squares of the coefficients.
- `abs(v)`: Return the Euclidean/L2-norm. Return the square root of `vecabsq`.
- `v @ w`: Return the real dot product of two vectors. No argument is complex conjugated. All coefficients are used as is.

vector space stuff
- `v + w`: Return the vector sum.
- `v - w`: Return the vector difference.
- `v * a`: Return the scalar product.
- `v / a`: Return the scalar true division.
- `v // a`: Return the scalar floor division.

## profiling

For most functions a performance comparison between different approaches has been made. The results can be found in [profiling.ipynb](profiling.ipynb).

## todo

 - [x] docstrings
 - [ ] `numpy` routines

## License (MIT)

Copyright (c) 2022-2024 Sebastian Gössl

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
