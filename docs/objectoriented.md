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

---

::: vector.objectoriented
    options:
      heading_level: 2
      members_order: source
      show_labels: true
    members: Vector
