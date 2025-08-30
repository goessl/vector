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

Padding is done with `int(0)`.

The functions are *type-independent*. However, the data types used must *support necessary scalar operations*. For instance, for vector addition, components must be addable — this may include operations with padded integer zeros. Empty operations return the zero vector (e.g. `vecadd()==veczero`) or integer zeros (e.g. `vecdot(veczero, veczero)==int(0)`).

---

## Creation

::: vector.functional
    options:
      members:
        - veczero
        - vecbasis
        - vecrand
        - vecrandn

## Utility

::: vector.functional
    options:
      members:
        - veceq
        - vectrim
        - vecround
        - vecrshift
        - veclshift

## Hilbert space

::: vector.functional
    options:
      members:
        - vecabsq
        - vecabs
        - vecdot
        - vecparallel

## Vector space

::: vector.functional
    options:
      members:
        - vecpos
        - vecneg
        - vecaddc
        - vecadd
        - vecsub
        - vecmul
        - vectruediv
        - vecfloordiv
        - vecmod
        - vecdivmod

## Elementwise

::: vector.functional
    options:
      members:
        - vechadamard
        - vechadamardtruediv
        - vechadamardfloordiv
        - vechadamardmod
        - vechadamardmin
        - vechadamardmax
