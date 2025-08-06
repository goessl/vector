# Object-oriented

```python
>>> from vector import Vector
>>> Vector((1, 2, 3))
Vector(1, 2, 3, ...)
>>> Vector.randn(3)
Vector(-0.5613820142699765, -0.028308921297709365, 0.8270724508948077, ...)
>>> Vector(3)
Vector(0, 0, 0, 1, ...)
```

The immutable `Vector` class wraps all the mentioned functions into a tidy package, making them easier to use by enabling interaction through operators.

Its coefficients are internally stored as a tuple in the `coef` attribute and therefore *zero-indexed*.

Vector operations return the same type (`type(v+w)==type(v)`) so the class can easily be extended (to e.g. a polynomial class).

## creation

- `Vector(i)`: Create a new vector with the given coefficients or the `i`-th basis vector if an integer `i` is given.

- `Vector.rand(n)`: Create a random vector of `n` uniform coefficients in `[0, 1[`.

- `Vector.randn(n, normed=True, mu=0, sigma=1))`: Create a random vector of `n` normal distributed coefficients.

- `Vector.ZERO`: Zero vector.

## sequence

- `len(v)`: Return the number of set coefficients.

- `v[key]`: Return the indexed coefficient or coefficients. Not set coefficients default to 0.

- `iter(v)`: Return an iterator over the set coefficients.

- `v == w`: Return if of same type with same coefficients.

- `v << i`: Return a vector with coefficients shifted to lower indices.

- `v >> i`: Return a vector with coefficients shifted to higher indices.

## utility

- `v.trim(tol=1e-9)`: Remove all trailing near zero (abs<=tol) coefficients.

- `v.round(ndigits=None)`: Round all coefficients to the given precision.

- `v.is_parallel(other)`: Return if the other vector is parallel.

## Hilbert space

- `v.absq()`: Return the sum of absolute squares of the coefficients.

- `abs(v)`: Return the Euclidean/L2-norm. Return the square root of `vecabsq`.

- `v @ w`: Return the inner product of two vectors without conjugation.

## vector space

- `+v`: Return the unary positive.

- `-v`: Return the negative.

- `.addc(c, i=0)`: Return the sum with the `i`-th basis vector times `c`.

- `v + w`: Return the vector sum.

- `v - w`: Return the vector difference.

- `v * a`: Return the scalar product.

- `v / a`: Return the scalar true division.

- `v // a`: Return the scalar floor division.

- `v % a`: Return the elementwise mod with a scalar.

## elementwise

- `v.hadamard(w)`: Return the elementwise product with another vector.

- `v.hadamardtruediv(w)`: Return the elementwise true division with another vector.

- `v.hadamardfloordiv(w)`: Return the elementwise floor division with another vector.

- `v.hadamardmod(w)`: Return the elementwise mod with another vector.

- `v.hadamardmin(w)`: Return the elementwise minimum with another vector.

- `v.hadamardmax(w)`: Return the elementwise maximum with another vector.
