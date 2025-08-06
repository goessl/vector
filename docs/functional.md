# Functions

```python
>>> from vector import vecadd
>>> a = (5, 6, 7)
>>> b = [2]
>>> c = range(4)
>>> vecadd(a, b, c)
(7, 7, 9, 3)
```

Prefixed by `vec...` (vector).

All functions *accept single exhaustible iterables*.

They *return vectors as tuples*.

The functions are *type-independent*. However, the data types used must *support necessary scalar operations*. For instance, for vector addition, components must be addable — this may include operations with padded integer zeros. Empty operations return the zero vector (e.g. `vecadd()==veczero`) or integer zeros (e.g. `vecdot(veczero, veczero)==int(0)`).

Padding is done with `int(0)`.

## creation

- `veczero = ()`: Zero vector.

- `vecbasis(i, c=1)`: Return the `i`-th basis vector times `c`. The returned value is a tuple with `i` integer zeros followed by `c`.

- `vecrand(n)`: Return a random vector of `n` uniform coefficients in `[0, 1[`.

- `vecrandn(n, normed=True, mu=0, sigma=1)`: Return a random vector of `n` normal distributed coefficients.

## utility

- `veceq(v, w)`: Return if two vectors are equal.

- `vectrim(v, tol=1e-9)`: Remove all trailing near zero (<=tol) coefficients.

- `vecround(v, ndigits=None)`: Round all coefficients to the given precision.

- `vecrshift(v, n)`: Pad `n` many zeros to the beginning of the vector.

- `veclshift(v, n)`: Remove `n` many coefficients at the beginning of the vector.

## Hilbert space

- `vecabsq(v)`: Return the sum of absolute squares of the coefficients.

- `vecabs(v)`: Return the Euclidean/L2-norm.

- `vecdot(v, w)`: Return the inner product of two vectors without conjugation.

- `vecparallel(v, w)`: Return if two vectors are parallel.

## vector space

- `vecpos(v)`: Return the vector with the unary positive operator applied.

- `vecneg(v)`: Return the vector with the unary negative operator applied.

- `vecaddc(v, c, i=0)`: Return `v` with `c` added to the `i`-th coefficient. More efficient than `vecadd(v, vecbasis(i, c)`.

- `vecadd(*vs)`: Return the sum of vectors.

- `vecsub(v, w)`: Return the difference of two vectors.

- `vecmul(a, v)`: Return the product of a scalar and a vector.

- `vectruediv(v, a)`: Return the true division of a vector and a scalar.

- `vecfloordiv(v, a)`: Return the floor division of a vector and a scalar.

- `vecmod(v, a)`: Return the elementwise mod of a vector and a scalar.

## elementwise

- `vechadamard(*vs)`: Return the elementwise product of vectors.

- `vechadamardtruediv(v, w)`: Return the elementwise true division of two vectors.

- `vechadamardfloordiv(v, w)`: Return the elementwise floor division of two vectors.

- `vechadamardmod(v, w)`: Return the elementwise mod of two vectors.

- `vechadamardmin(*vs)`: Return the elementwise minimum of vectors.

- `vechadamardmax(*vs)`: Return the elementwise maximum of vectors.
