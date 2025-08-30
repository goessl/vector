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

---

## Creation

::: vector.parallelised
    options:
      members:
        - vecnpzero
        - vecnpbasis
        - vecnprand
        - vecnprandn

## Utility

::: vector.parallelised
    options:
      members:
        - vecnpeq
        - vecnptrim
        - vecnpround

## Hilbert space

::: vector.parallelised
    options:
      members:
        - vecnpabsq
        - vecnpabs
        - vecnpdot
        - vecnpparallel

## Vector space

::: vector.parallelised
    options:
      members:
        - vecnppos
        - vecnpneg
        - vecnpadd
        - vecnpsub
        - vecnpmul
        - vecnptruediv
        - vecnpfloordiv
        - vecnpmod
