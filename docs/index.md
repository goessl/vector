# vector

An infinite-dimensional vector Python package.
```python
>>> from vector import vecadd
>>> vecadd((1, 2), (4, 5, 6))
(5, 7, 6)
>>> 
>>> from vector import Vector
>>> v = Vector((1, 2))
>>> w = Vector((4, 5, 6))
>>> v + w
Vector(5, 7, 6, ...)
>>> 
>>> from vector import vecnpadd
>>> vecnpadd((1, 2), ((3, 4, 5),
...                   (6, 7, 8)))
array([[4, 6, 5],
       [7, 9, 8]])
```

## Installation

```console
pip install git+https://github.com/goessl/vector.git
```

## Usage

This package includes

- general-purpose **functions** (prefixed `vec...`) in *pure Python*,
- a clean **class** (`Vector`) with *easy to use syntax*,
- improved *numpy-routines* (prefixed `vecnp...`) for **parallelised
operations** &
- *tensor* functions (prefixed `ten...`) for **multiaxis operations**

to handle **type-independent, infinite-dimensional** vectors.
It operates on vectors of different lengths, treating them as
infinite-dimensional by assuming that all components after the given ones are
*zero*.

| Operation         | [Functional](functional.md)                                    | [Object-oriented](objectoriented.md)                                 | [Parallelised](parallelised.md)                      | [Multiaxis](multiaxis.md)                                     |
| ----------------- | -------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| **Creation**      |                                                                |                                                                      |                                                      |                                                               |
| Zero constant     | [`veczero`][vector.functional.veczero]                         | [`Vector.ZERO`][vector.objectoriented.Vector.ZERO]                   | [`vecnpzero`][vector.parallelised.vecnpzero]         | [`tenzero`][vector.multiaxis.tenzero]                         |
| Basis             | [`vecbasis`][vector.functional.vecbasis]                       | [`Vector`][vector.objectoriented.Vector]                             | [`vecnpbasis`][vector.parallelised.vecnpbasis]       | [`tenbasis`][vector.multiaxis.tenbasis]                       |
| Random uniform    | [`vecrand`][vector.functional.vecrand]                         | [`Vector.rand`][vector.objectoriented.Vector.rand]                   | [`vecnprand`][vector.parallelised.vecnprand]         | [`tenrand`][vector.multiaxis.tenrand]                         |
| Random normal     | [`vecrandn`][vector.functional.vecrandn]                       | [`Vector.randn`][vector.objectoriented.Vector.randn]                 | [`vecnprandn`][vector.parallelised.vecnprandn]       | [`tenrandn`][vector.multiaxis.tenrandn]                       |
| **Utility**       |                                                                |                                                                      |                                                      |                                                               |
| Dimensionality    |                                                                | [`len`][vector.objectoriented.Vector.__len__]                        | [`vecnpdim`][vector.parallelised.vecnpdim]           | [`tendim`][vector.multiaxis.tendim]                           |
| Rank              |                                                                |                                                                      |                                                      | [`tenrank`][vector.multiaxis.tenrank]                         |
| Trimming          | [`vectrim`][vector.functional.vectrim]                         | [`.trim`][vector.objectoriented.Vector.trim]                         | [`vecnptrim`][vector.parallelised.vecnptrim]         | [`tentrim`][vector.multiaxis.tentrim]                         |
| Comparison        | [`veceq`][vector.functional.veceq]                             | [`==`][vector.objectoriented.Vector.__eq__]                          | [`vecnpeq`][vector.parallelised.vecnpeq]             |                                                               |
| Rounding          | [`vecround`][vector.functional.vecround]                       | [`.round`][vector.objectoriented.Vector.round]                       | [`vecnpround`][vector.parallelised.vecnpround]       | [`tenround`][vector.multiaxis.tenround]                       |
| Right shift       | [`vecrshift`][vector.functional.vecrshift]                     | [`>>`][vector.objectoriented.Vector.__rshift__]                      |                                                      |                                                               |
| Left shift        | [`veclshift`][vector.functional.veclshift]                     | [`<<`][vector.objectoriented.Vector.__lshift__]                      |                                                      |                                                               |
| **Hilbert space** |                                                                |                                                                      |                                                      |                                                               |
| Norm              | [`vecabs`][vector.functional.vecabs]                           | [`abs`][vector.objectoriented.Vector.__abs__]                        | [`vecnpabs`][vector.parallelised.vecnpabs]           |                                                               |
| Norm squared      | [`vecabsq`][vector.functional.vecabsq]                         | [`.absq`][vector.objectoriented.Vector.absq]                         | [`vecnpabsq`][vector.parallelised.vecnpabsq]         |                                                               |
| Inner product     | [`vecdot`][vector.functional.vecdot]                           | [`@`][vector.objectoriented.Vector.__matmul__]                       | [`vecnpdot`][vector.parallelised.vecnpdot]           |                                                               |
| Parallelism       | [`vecparallel`][vector.functional.vecparallel]                 | [`.is_parallel`][vector.objectoriented.Vector.is_parallel]           | [`vecnpparallel`][vector.parallelised.vecnpparallel] |                                                               |
| **Vector space**  |                                                                |                                                                      |                                                      |                                                               |
| Positive          | [`vecpos`][vector.functional.vecpos]                           | [`+`][vector.objectoriented.Vector.__pos__]                          | [`vecnppos`][vector.parallelised.vecnppos]           | [`tenpos`][vector.multiaxis.tenpos]                           |
| Negative          | [`vecneg`][vector.functional.vecneg]                           | [`-`][vector.objectoriented.Vector.__neg__]                          | [`vecnpneg`][vector.parallelised.vecnpneg]           | [`tenneg`][vector.multiaxis.tenneg]                           |
| Addition          | [`vecadd`][vector.functional.vecadd]                           | [`+`][vector.objectoriented.Vector.__add__]                          | [`vecnpadd`][vector.parallelised.vecnpadd]           | [`tenadd`][vector.multiaxis.tenadd]                           |
| Basis addition    | [`vecaddc`][vector.functional.vecaddc]                         | [`.addc`][vector.objectoriented.Vector.addc]                         |                                                      | [`tenaddc`][vector.multiaxis.tenaddc]                         |
| Subtraction       | [`vecsub`][vector.functional.vecsub]                           | [`-`][vector.objectoriented.Vector.__sub__]                          | [`vecnpsub`][vector.parallelised.vecnpsub]           | [`tensub`][vector.multiaxis.tensub]                           |
| Multiplication    | [`vecmul`][vector.functional.vecmul]                           | [`*`][vector.objectoriented.Vector.__mul__]                          | [`vecnpmul`][vector.parallelised.vecnpmul]           | [`tenmul`][vector.multiaxis.tenmul]                           |
| True division     | [`vectruediv`][vector.functional.vectruediv]                   | [`/`][vector.objectoriented.Vector.__truediv__]                      | [`vecnptruediv`][vector.parallelised.vecnptruediv]   | [`tentruediv`][vector.multiaxis.tentruediv]                   |
| Floor division    | [`vecfloordiv`][vector.functional.vecfloordiv]                 | [`//`][vector.objectoriented.Vector.__floordiv__]                    | [`vecnpfloordiv`][vector.parallelised.vecnpfloordiv] | [`tenfloordiv`][vector.multiaxis.tenfloordiv]                 |
| Mod               | [`vecmod`][vector.functional.vecmod]                           | [`%`][vector.objectoriented.Vector.__mod__]                          | [`vecnpmod`][vector.parallelised.vecnpmod]           | [`tenmod`][vector.multiaxis.tenmod]                           |
| Divmod            | [`vecdivmod`][vector.functional.vecdivmod]                     | [`divmod`][vector.objectoriented.Vector.__divmod__]                  |                                                      | [`tendivmod`][vector.multiaxis.tendivmod]                     |
| **Elementwise**   |                                                                |                                                                      |                                                      |                                                               |
| Multiplication    | [`vechadamard`][vector.functional.vechadamard]                 | [`.hadamard`][vector.objectoriented.Vector.hadamard]                 |                                                      | [`tenhadamard`][vector.multiaxis.tenhadamard]                 |
| True division     | [`vechadamardtruediv`][vector.functional.vechadamardtruediv]   | [`.hadamardtruediv`][vector.objectoriented.Vector.hadamardtruediv]   |                                                      | [`tenhadamardtruediv`][vector.multiaxis.tenhadamardtruediv]   |
| Floor division    | [`vechadamardfloordiv`][vector.functional.vechadamardfloordiv] | [`.hadamardfloordiv`][vector.objectoriented.Vector.hadamardfloordiv] |                                                      | [`tenhadamardfloordiv`][vector.multiaxis.tenhadamardfloordiv] |
| Mod               | [`vechadamardmod`][vector.functional.vechadamardmod]           | [`.hadamardmod`][vector.objectoriented.Vector.hadamardmod]           |                                                      | [`tenhadamardmod`][vector.multiaxis.tenhadamardmod]           |
| Min               | [`vechadamardmin`][vector.functional.vechadamardmin]           | [`.hadamardmin`][vector.objectoriented.Vector.hadamardmin]           |                                                      |                                                               |
| Max               | [`vechadamardmax`][vector.functional.vechadamardmax]           | [`.hadamardmax`][vector.objectoriented.Vector.hadamardmax]           |                                                      |                                                               |

### Prefix Design

Could use no prefix to be more mathematically pure, like `add` instead of
`vecadd`, but then you would always have to use `from vec import add as vecadd`
if used with other libraries (like `operator`).

Also avoids keyword collisions (`abs` is reserved, `vecabs` isn't).

Do it like `numpy.polynomial.polynomial. ...`.

## Roadmap

- [x] `zip` version between `zip` & `zip_longest`. Yields different sized
tuples. Done: [goessl/zipvar](https://github.com/goessl/zipvar)
- [x] `vecdivmod`
- [x] docstrings
- [x] `numpy` routines
- [x] multiaxis vectors: tensors?
- [ ] Complexity analysis. Perfect complexity
- [ ] dimensionality signature (e.g. `vecadd`: $\mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max{m, n}}$)
- [ ] `vechadamardminmax`
- [ ] never use `numpy.int64`, they don't detect overflows
- [ ] sparse vectors (`dict`s)
- [ ] C++ & Java version

## License (MIT)

Copyright (c) 2022-2025 Sebastian Gössl

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
