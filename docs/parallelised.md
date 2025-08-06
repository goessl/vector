# Parallelised

```python
>>> from vector import vecnpadd
>>> vecnpadd((1, 2), ((3, 4, 5),
...                   (6, 7, 8)))
array([[4, 6, 5],
       [7, 9, 8]])
```

Prefixed by `vecnp...` (vector numpy).

`numpy`-versions of the functions are also provided, to *operate on multiple vectors* at once. They behave like the ones in `numpy.polynomial.polynomial`, but *also work on 2D-arrays* (and *all combinations of 1D & 2D arrays*) and broadcast to multiple dimensions like the usual `numpy` operations (but adjust the shapes accordingly).

*`vecnpzero` is `np.array([0])`* like `numpy.polynomial.polynomial.polyzero`, not `veczero=()` (empty tuple, no zero coefficient left) like in the functions and class above.

Padding is done with `numpy.int64(0)`.

They return scalars or `numpy.ndarray`s.

Creation routines have a dimension argument `d`. If left to `None`, the returned values are 1D, so a single vector. If given, the routines return a 2D-array representing multiple vectors in rows.

## creation

- `vecnpzero(d=None)`: Return `d` zero vectors. The returned value is a `(d, 1)`-array of zeros if `d` is not `None` or `[0]` otherwise.

- `vecnpbasis(i, c=1, d=None)`: Return `d` many `i`-th basis vectors times `c`. The returned value is a `(d, i+1)`-array if `d` is not `None` or `(i+1,)` otherwise.

- `vecnprand(n, d=None)`: Return `d` random vectors of `n` uniform coefficients in `[0, 1[`. The returned value is a `(d, n)`-array if `d` is not `None` or `(n,)` otherwise.

- `vecnprandn(n, normed=True, d=None)`: Return `d` random vectors of `n` normal distributed coefficients. The returned value is a `(d, n)`-array if `d` is not `None` or `(n,)` otherwise.

## utility

- `vecnpeq(v, w)`: Return if two vectors are equal.

- `vecnptrim(v, tol=1e-9)`: Remove all trailing near zero (abs(v_i)<=tol) coefficients.

- `vecnpround(v, ndigits=0)`: Wrapper for `numpy.round`.

## Hilbert space

- `vecnpabsq(v)`: Return the sum of absolute squares of the coefficients.

- `vecnpabs(v)`: Return the Euclidean/L2-norm.

- `vecnpdot(v, w)`: Return the inner product of two vectors without conjugation.

- `vecnpparallel(v, w)`: Return if two vectors are parallel.

## vector space

- `vecnppos(v)`: Return the vector with the unary positive operator applied.

- `vecnpneg(v)`: Return the vector with the unary negative operator applied.

- `vecnpadd(*vs)`: Return the sum of vectors.

- `vecnpsub(v, w)`: Return the difference of two vectors.

- `vecnpmul(a, v)`: Return the product of a scalar and a vector.

- `vecnptruediv(v, a)`: Return the true division of a vector and a scalar.

- `vecnpfloordiv(v, a)`: Return the floor division of a vector and a scalar.

- `vecnpmod(v, a)`: Return the elementwise mod of a vector and a scalar.
