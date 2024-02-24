# vector

An infinite dimensional vector package.
```python
>>> from vector import Vector
>>> v = Vector((1, 2))
>>> w = Vector((4, 5, 6))
>>> v+w
Vector(5, 7, 6, 0, ...)
```

## Installation

```console
pip install git+https://github.com/goessl/vector.git
```

## Usage

This package provides a single class, `Vector`, to handle infinite dimensional vectors.
A vector can be initialized in two ways:
 - With the constructor `Vector(coef)`, that takes a non-negative integer to create a basis vector with the given index, or an iterable of coefficients to create a vector with the given coefficients as the first elements.
 - With the random factory `Vector.random(n, normed=True)` for a random vector of a given dimensionality.
A static helper method `Vector.basis_tuple(i)` is also provided, that generates a basis vector in form of a tuple.
The objects are immutable (coefficients are internally stored in a tuple) and zero-indexed.
```python
>>> from vector import Vector
>>> v = Vector((1, 2, 3))
>>> w = Vector.random(3)
>>> v
Vector(1, 2, 3, 0, ...)
>>> w
Vector(-0.5613820142699765, -0.028308921297709365, 0.8270724508948077, 0, ...)
>>> Vector.basis_tuple(3)
(0, 0, 0, 1)
```

Container and sequence interfaces are implemented so the coefficients can be
- accessed by indexing: `v[2]` (coefficients not set return to 0),
- iterated over: `for c in v` (stops at last set coefficient),
- counted: `len(v)` (number of set coefficients),
- compared: `v == w` (tuple of coefficients get compared),
- shifted: `v >> 1, w << 2` &
- trimmed: `v.trim()` (trailing non-zero coefficients get removed).

Hilbert space operations are provided:
- Vector addition & subtraction `v + w, v - w`,
- scalar multiplication & division `2 * v, w / 2`,
- inner product & norm `v @ w, abs(v)` (real inner product; complex conjugation of an argument has to be handled by the user; to comply with [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)).

The multiplicative operations are overloaded to perform scalar multiplication/division if the other argument is a scalar, or elementwise multiplication/division if both operands are `Vector`s, `v*w`.

A static zero-vector `Vector.ZERO` is provided.

## profiling

For most methods a runtime comparison between different approaches has been made. The results can be found in [profiling.ipynb](profiling.ipynb) or [profiling.ipynb](profiling.pdf).

## License (MIT)

Copyright (c) 2023 Sebastian Gössl

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
