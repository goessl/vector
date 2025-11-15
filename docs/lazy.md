# Lazy

```python
>>> from vector import veclbasis
>>> veclbasis(3)
<generator object veclbasis at 0x0123456789ABCDEF>
>>> tuple(veclbasis(3))
(0, 0, 0, 1)
```

Prefixed by `vecl...` (vector - lazy).

Lazy generator versions of [`functional`](functional.md).

Different behaviour:

- [`vecrandn`][vector.functional.vecrandn]: normalisation not possible.

Lazy versions not possible/necessary as the don't return vectors:

- [`vecabs`][vector.functional.vecabs]
- [`vecabsq`][vector.functional.vecabsq]
- [`vecdot`][vector.functional.vecdot]
- [`vecparallel`][vector.functional.vecparallel]

---

## Creation

::: vector.lazy
    options:
      members:
        - veclzero
        - veclbasis
        - veclbases
        - veclrand
        - veclrandn

## Utility

::: vector.lazy
    options:
      members:
        - vecleq
        - vecltrim
        - veclround
        - veclrshift
        - vecllshift

## Hilbert space

::: vector.lazy
    options:
      members:
        - try_conugate
        - veclconj

## Vector space

::: vector.lazy
    options:
      members:
        - veclpos
        - veclneg
        - vecladd
        - vecladdc
        - veclsub
        - veclsubc
        - veclmul
        - vecltruediv
        - veclfloordiv
        - veclmod
        - vecldivmod

## Elementwise

::: vector.lazy
    options:
      members:
        - veclhadamard
        - veclhadamardtruediv
        - veclhadamardfloordiv
        - veclhadamardmod
        - veclhadamarddivmod
        - veclhadamardmin
        - veclhadamardmax
