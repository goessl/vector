# Multiaxis

Prefixed by `ten...` (tensor).

Handle multiaxis vectors, that for example represent multivariate polynomials.

Tensors are returned as `numpy.ndarray`s.

Broadcasting happens similar to [`numpy`s broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html),
but the axes are matched in ascending order instead of descending order, and
the arrays don't get stretched but rather padded with zeros.

---

## Creation

::: vector.multiaxis
    options:
      heading_level: 3
      members:
        - tenzero
        - tenbasis
        - tenrand
        - tenrandn

## Utility

::: vector.multiaxis
    options:
      heading_level: 3
      members:
        - tenrank
        - tendim
        - tentrim
        - tenround

## Vector space

::: vector.multiaxis
    options:
      heading_level: 3
      members:
        - tenpos
        - tenneg
        - tenaddc
        - tenadd
        - tensub
        - tenmul
        - tentruediv
        - tenfloordiv
        - tenmod
        - tendivmod

## Elementwise

::: vector.multiaxis
    options:
      heading_level: 3
      members:
        - tenhadamard
        - tenhadamardtruediv
        - tenhadamardfloordiv
        - tenhadamardmod
