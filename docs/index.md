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

- general-purpose **functions** (prefixed `vec...`) in *pure Python* with **perfect complexity**,
- lazy **generators** (prefixed `vecl...`),
- **in-place** modifications (prefixed `veci...`),
- `dicts` as **sparse** vectors (prefixed `vecs...`),
- *tensor* functions (prefixed `ten...`) for **multilinear operations**,
- sparse *tensor* functions (prefixed `tens...`) for **multilinear sparse operations** &
- improved *numpy-routines* (prefixed `vecnp...`) for **parallelised
operations**.

Functional, sparse and multilinear sparse additionally contain wrapper classes.

to handle **type-independent, infinite-dimensional** vectors.
It operates on vectors of different lengths, treating them as
infinite-dimensional by assuming that all components after the given ones are
*zero*.

All vectors are **zero-indexed**.

| Operation         | [Functional](functional.md)                                                | [Lazy](lazy.md)                                                        | [In-place](inplace.md)                                                    | [Sparse](sparse.md)                                                      | [Multilinear](multilinear.md)                                               | [Multilinear sparse](multilinear_sparse.md)                                          | [Parallelised](parallelised.md)                      |
| ----------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| **Creation**      |                                                                            |                                                                        |                                                                           |                                                                          |                                                                             |                                                                                      |                                                      |
| Zero constant     | [`veczero`][vector.functional.creation.veczero]                            | [`veclzero`][vector.lazy.creation.veclzero]                            | [`vecizero`][vector.inplace.creation.vecizero]                            | [`vecszero`][vector.sparse.creation.vecszero]                            | [`tenzero`][vector.multilinear.creation.tenzero]                            | [`tenszero`][vector.multilinear_sparse.creation.tenszero]                            | [`vecnpzero`][vector.parallelised.vecnpzero]         |
| Basis             | [`vecbasis`][vector.functional.creation.vecbasis]                          | [`veclbasis`][vector.lazy.creation.veclbasis]                          | [`vecibasis`][vector.inplace.creation.vecibasis]                          | [`vecsbasis`][vector.sparse.creation.vecsbasis]                          | [`tenbasis`][vector.multilinear.creation.tenbasis]                          | [`tensbasis`][vector.multilinear_sparse.creation.tensbasis]                          | [`vecnpbasis`][vector.parallelised.vecnpbasis]       |
| Bases             | [`vecbases`][vector.functional.creation.vecbases]                          | [`veclbases`][vector.lazy.creation.veclbases]                          | [`vecibases`][vector.inplace.creation.vecibases]                          | [`vecsbases`][vector.sparse.creation.vecsbases]                          |                                                                             |                                                                                      |                                                      |
| Random uniform    | [`vecrand`][vector.functional.creation.vecrand]                            | [`veclrand`][vector.lazy.creation.veclrand]                            | [`vecirand`][vector.inplace.creation.vecirand]                            | [`vecsrand`][vector.sparse.creation.vecsrand]                            | [`tenrand`][vector.multilinear.creation.tenrand]                            | [`tensrand`][vector.multilinear_sparse.creation.tensrand]                            | [`vecnprand`][vector.parallelised.vecnprand]         |
| Random normal     | [`vecrandn`][vector.functional.creation.vecrandn]                          | [`veclrandn`][vector.lazy.creation.veclrandn]                          | [`vecirandn`][vector.inplace.creation.vecirandn]                          | [`vecsrandn`][vector.sparse.creation.vecsrandn]                          | [`tenrandn`][vector.multilinear.creation.tenrandn]                          | [`tensrandn`][vector.multilinear_sparse.creation.tensrandn]                          | [`vecnprandn`][vector.parallelised.vecnprandn]       |
| **Conversion**    |                                                                            |                                                                        |                                                                           |                                                                          |                                                                             |                                                                                      |                                                      |
| to special type   |                                                                            |                                                                        |                                                                           | [`vecdtos`][vector.sparse.conversion.vecdtos]                            |                                                                             |                                                                                      |                                                      |
| from special type |                                                                            |                                                                        |                                                                           | [`vecstod`][vector.sparse.conversion.vecstod]                            |                                                                             |                                                                                      |                                                      |
| **Utility**       |                                                                            |                                                                        |                                                                           |                                                                          |                                                                             |                                                                                      |                                                      |
| Dimensionality    | [`veclen`][vector.functional.utility.veclen]                               |                                                                        |                                                                           |                                                                          | [`tendim`][vector.multilinear.utility.tendim]                               | [`tensdim`][vector.multilinear_sparse.utility.tensdim]                               | [`vecnpdim`][vector.parallelised.vecnpdim]           |
| Rank              |                                                                            |                                                                        |                                                                           | [`vecslen`][vector.sparse.utility.vecslen]                               | [`tenrank`][vector.multilinear.utility.tenrank]                             | [`tensrank`][vector.multilinear_sparse.utility.tensrank]                             |                                                      |
| Comparison        | [`veceq`][vector.functional.utility.veceq]                                 | [`vecleq`][vector.lazy.utility.vecleq]                                 |                                                                           | [`vecseq`][vector.sparse.utility.vecseq]                                 |                                                                             | [`tenseq`][vector.multilinear_sparse.utility.tenseq]                                 | [`vecnpeq`][vector.parallelised.vecnpeq]             |
| Trimming          | [`vectrim`][vector.functional.utility.vectrim]                             | [`vecltrim`][vector.lazy.utility.vecltrim]                             | [`vecitrim`][vector.inplace.utility.vecitrim]                             | [`vecstrim`][vector.sparse.utility.vecstrim]                             | [`tentrim`][vector.multilinear.utility.tentrim]                             | [`tenstrim`][vector.multilinear_sparse.utility.tenstrim]                             | [`vecnptrim`][vector.parallelised.vecnptrim]         |
| Right shift       | [`vecrshift`][vector.functional.utility.vecrshift]                         | [`veclrshift`][vector.lazy.utility.veclrshift]                         | [`vecirshift`][vector.inplace.utility.vecirshift]                         | [`vecsrshift`][vector.sparse.utility.vecsrshift]                         | [`tenrshift`][vector.multilinear.utility.tenrshift]                         | [`tensrshift`][vector.multilinear_sparse.utility.tensrshift]                         |                                                      |
| Left shift        | [`veclshift`][vector.functional.utility.veclshift]                         | [`vecllshift`][vector.lazy.utility.vecllshift]                         | [`vecilshift`][vector.inplace.utility.vecilshift]                         | [`vecslshift`][vector.sparse.utility.vecslshift]                         | [`tenlshift`][vector.multilinear.utility.tenlshift]                         | [`tenslshift`][vector.multilinear_sparse.utility.tenslshift]                         |                                                      |
| **Hilbert space** |                                                                            |                                                                        |                                                                           |                                                                          |                                                                             |                                                                                      |                                                      |
| Conjugation       | [`vecconj`][vector.functional.hilbert_space.vecconj]                       | [`veclconj`][vector.lazy.hilbert_space.veclconj]                       | [`veciconj`][vector.inplace.hilbert_space.veciconj]                       | [`vecsconj`][vector.sparse.hilbert_space.vecsconj]                       | [`tenconj`][vector.multilinear.hilbert_space.tenconj]                       | [`tensconj`][vector.multilinear_sparse.hilbert_space.tensconj]                       |                                                      |
| Norm              | [`vecabs`][vector.functional.hilbert_space.vecabs]                         |                                                                        |                                                                           | [`vecsabs`][vector.sparse.hilbert_space.vecsabs]                         |                                                                             |                                                                                      | [`vecnpabs`][vector.parallelised.vecnpabs]           |
| Norm squared      | [`vecabsq`][vector.functional.hilbert_space.vecabsq]                       |                                                                        |                                                                           | [`vecsabsq`][vector.sparse.hilbert_space.vecsabsq]                       |                                                                             |                                                                                      | [`vecnpabsq`][vector.parallelised.vecnpabsq]         |
| Inner product     | [`vecdot`][vector.functional.hilbert_space.vecdot]                         |                                                                        |                                                                           | [`vecsdot`][vector.sparse.hilbert_space.vecsdot]                         |                                                                             |                                                                                      | [`vecnpdot`][vector.parallelised.vecnpdot]           |
| Parallelism       | [`vecparallel`][vector.functional.hilbert_space.vecparallel]               |                                                                        |                                                                           | [`vecsparallel`][vector.sparse.hilbert_space.vecsparallel]               |                                                                             |                                                                                      | [`vecnpparallel`][vector.parallelised.vecnpparallel] |
| **Vector space**  |                                                                            |                                                                        |                                                                           |                                                                          |                                                                             |                                                                                      |                                                      |
| Positive          | [`vecpos`][vector.functional.vector_space.vecpos]                          | [`veclpos`][vector.lazy.vector_space.veclpos]                          | [`vecipos`][vector.inplace.vector_space.vecipos]                          | [`vecspos`][vector.sparse.vector_space.vecspos]                          | [`tenpos`][vector.multilinear.vector_space.tenpos]                          | [`tenspos`][vector.multilinear_sparse.vector_space.tenspos]                          | [`vecnppos`][vector.parallelised.vecnppos]           |
| Negative          | [`vecneg`][vector.functional.vector_space.vecneg]                          | [`veclneg`][vector.lazy.vector_space.veclneg]                          | [`vecineg`][vector.inplace.vector_space.vecineg]                          | [`vecsneg`][vector.sparse.vector_space.vecsneg]                          | [`tenneg`][vector.multilinear.vector_space.tenneg]                          | [`tensneg`][vector.multilinear_sparse.vector_space.tensneg]                          | [`vecnpneg`][vector.parallelised.vecnpneg]           |
| Addition          | [`vecadd`][vector.functional.vector_space.vecadd]                          | [`vecladd`][vector.lazy.vector_space.vecladd]                          | [`veciadd`][vector.inplace.vector_space.veciadd]                          | [`vecsadd`][vector.sparse.vector_space.vecsadd]                          | [`tenadd`][vector.multilinear.vector_space.tenadd]                          | [`tensadd`][vector.multilinear_sparse.vector_space.tensadd]                          | [`vecnpadd`][vector.parallelised.vecnpadd]           |
| Basis addition    | [`vecaddc`][vector.functional.vector_space.vecaddc]                        | [`vecladdc`][vector.lazy.vector_space.vecladdc]                        | [`veciaddc`][vector.inplace.vector_space.veciaddc]                        | [`vecsaddc`][vector.sparse.vector_space.vecsaddc]                        | [`tenaddc`][vector.multilinear.vector_space.tenaddc]                        | [`tensaddc`][vector.multilinear_sparse.vector_space.tensaddc]                        |                                                      |
| Subtraction       | [`vecsub`][vector.functional.vector_space.vecsub]                          | [`veclsub`][vector.lazy.vector_space.veclsub]                          | [`vecisub`][vector.inplace.vector_space.vecisub]                          | [`vecssub`][vector.sparse.vector_space.vecssub]                          | [`tensub`][vector.multilinear.vector_space.tensub]                          | [`tenssub`][vector.multilinear_sparse.vector_space.tenssub]                          | [`vecnpsub`][vector.parallelised.vecnpsub]           |
| Basis subtraction | [`vecsubc`][vector.functional.vector_space.vecsubc]                        | [`veclsubc`][vector.lazy.vector_space.veclsubc]                        | [`vecisubc`][vector.inplace.vector_space.vecisubc]                        | [`vecssubc`][vector.sparse.vector_space.vecssubc]                        | [`tensubc`][vector.multilinear.vector_space.tensubc]                        | [`tenssubc`][vector.multilinear_sparse.vector_space.tenssubc]                        |                                                      |
| Multiplication    | [`vecmul`][vector.functional.vector_space.vecmul]                          | [`veclmul`][vector.lazy.vector_space.veclmul]                          | [`vecimul`][vector.inplace.vector_space.vecimul]                          | [`vecsmul`][vector.sparse.vector_space.vecsmul]                          | [`tenmul`][vector.multilinear.vector_space.tenmul]                          | [`tensmul`][vector.multilinear_sparse.vector_space.tensmul]                          | [`vecnpmul`][vector.parallelised.vecnpmul]           |
| True division     | [`vectruediv`][vector.functional.vector_space.vectruediv]                  | [`vecltruediv`][vector.lazy.vector_space.vecltruediv]                  | [`vecitruediv`][vector.inplace.vector_space.vecitruediv]                  | [`vecstruediv`][vector.sparse.vector_space.vecstruediv]                  | [`tentruediv`][vector.multilinear.vector_space.tentruediv]                  | [`tenstruediv`][vector.multilinear_sparse.vector_space.tenstruediv]                  | [`vecnptruediv`][vector.parallelised.vecnptruediv]   |
| Floor division    | [`vecfloordiv`][vector.functional.vector_space.vecfloordiv]                | [`veclfloordiv`][vector.lazy.vector_space.veclfloordiv]                | [`vecifloordiv`][vector.inplace.vector_space.vecifloordiv]                | [`vecsfloordiv`][vector.sparse.vector_space.vecsfloordiv]                | [`tenfloordiv`][vector.multilinear.vector_space.tenfloordiv]                | [`tensfloordiv`][vector.multilinear_sparse.vector_space.tensfloordiv]                | [`vecnpfloordiv`][vector.parallelised.vecnpfloordiv] |
| Mod               | [`vecmod`][vector.functional.vector_space.vecmod]                          | [`veclmod`][vector.lazy.vector_space.veclmod]                          | [`vecimod`][vector.inplace.vector_space.vecimod]                          | [`vecstruediv`][vector.sparse.vector_space.vecsmod]                      | [`tenmod`][vector.multilinear.vector_space.tenmod]                          | [`tensmod`][vector.multilinear_sparse.vector_space.tensmod]                          | [`vecnpmod`][vector.parallelised.vecnpmod]           |
| Divmod            | [`vecdivmod`][vector.functional.vector_space.vecdivmod]                    | [`vecldivmod`][vector.lazy.vector_space.vecldivmod]                    |                                                                           | [`vecsdivmod`][vector.sparse.vector_space.vecsdivmod]                    | [`tendivmod`][vector.multilinear.vector_space.tendivmod]                    | [`tensdivmod`][vector.multilinear_sparse.vector_space.tensdivmod]                    |                                                      |
| **Elementwise**   |                                                                            |                                                                        |                                                                           |                                                                          |                                                                             |                                                                                      |                                                      |
| Multiplication    | [`vechadamard`][vector.functional.elementwise.vechadamard]                 | [`veclhadamard`][vector.lazy.elementwise.veclhadamard]                 | [`vecihadamard`][vector.inplace.elementwise.vecihadamard]                 | [`vecshadamard`][vector.sparse.elementwise.vecshadamard]                 | [`tenhadamard`][vector.multilinear.elementwise.tenhadamard]                 | [`tenshadamard`][vector.multilinear_sparse.elementwise.tenshadamard]                 |                                                      |
| True division     | [`vechadamardtruediv`][vector.functional.elementwise.vechadamardtruediv]   | [`veclhadamardtruediv`][vector.lazy.elementwise.veclhadamardtruediv]   | [`vecihadamardtruediv`][vector.inplace.elementwise.vecihadamardtruediv]   | [`vecshadamardtruediv`][vector.sparse.elementwise.vecshadamardtruediv]   | [`tenhadamardtruediv`][vector.multilinear.elementwise.tenhadamardtruediv]   | [`tenshadamardtruediv`][vector.multilinear_sparse.elementwise.tenshadamardtruediv]   |                                                      |
| Floor division    | [`vechadamardfloordiv`][vector.functional.elementwise.vechadamardfloordiv] | [`veclhadamardfloordiv`][vector.lazy.elementwise.veclhadamardfloordiv] | [`vecihadamardfloordiv`][vector.inplace.elementwise.vecihadamardfloordiv] | [`vecshadamardfloordiv`][vector.sparse.elementwise.vecshadamardfloordiv] | [`tenhadamardfloordiv`][vector.multilinear.elementwise.tenhadamardfloordiv] | [`tenshadamardfloordiv`][vector.multilinear_sparse.elementwise.tenshadamardfloordiv] |                                                      |
| Mod               | [`vechadamardmod`][vector.functional.elementwise.vechadamardmod]           | [`veclhadamardmod`][vector.lazy.elementwise.veclhadamardmod]           | [`vecihadamardmod`][vector.inplace.elementwise.vecihadamardmod]           | [`vecshadamardmod`][vector.sparse.elementwise.vecshadamardmod]           | [`tenhadamardmod`][vector.multilinear.elementwise.tenhadamardmod]           | [`tenshadamardmod`][vector.multilinear_sparse.elementwise.tenshadamardmod]           |                                                      |
| Divmod            | [`vechadamarddivmod`][vector.functional.elementwise.vechadamarddivmod]     | [`veclhadamarddivmod`][vector.lazy.elementwise.veclhadamarddivmod]     |                                                                           | [`vecshadamarddivmod`][vector.sparse.elementwise.vecshadamarddivmod]     | [`tenhadamarddivmod`][vector.multilinear.elementwise.tenhadamarddivmod]     | [`tenshadamarddivmod`][vector.multilinear_sparse.elementwise.tenshadamarddivmod]     |                                                      |
| Min               | [`vechadamardmin`][vector.functional.elementwise.vechadamardmin]           | [`veclhadamardmin`][vector.lazy.elementwise.veclhadamardmin]           |                                                                           | [`vecshadamardmin`][vector.sparse.elementwise.vecshadamardmin]           | [`tenhadamardmin`][vector.multilinear.elementwise.tenhadamardmin]           | [`tenshadamardmin`][vector.multilinear_sparse.elementwise.tenshadamardmin]           |                                                      |
| Max               | [`vechadamardmax`][vector.functional.elementwise.vechadamardmax]           | [`veclhadamardmax`][vector.lazy.elementwise.veclhadamardmax]           |                                                                           | [`vecshadamardmax`][vector.sparse.elementwise.vecshadamardmax]           | [`tenhadamardmax`][vector.multilinear.elementwise.tenhadamardmax]           | [`tenshadamardmax`][vector.multilinear_sparse.elementwise.tenshadamardmax]           |                                                      |

### Design choices

1. Integers are the best.
   As many functions as possible should work with pure integer arithmetic.
2. Floats are necessary. (Also let's don't forget about complex numbers.)
   When possible, extended precision intermediates are used (`sum`, `sumprod`, ...)
3. Python allows operator overloading.
   Exclusive type arithmetic should be possible (`zero`, `one` & `inf` arguments; ...)

#### Prefix Design

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
- [x] multilinear vectors: tensors?
- [x] Absolute type safety.
- [x] Complexity analysis. Perfect complexity
- [x] dimensionality signature (e.g. `vecadd`: $\mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{\max{m, n}}$)
- [x] sparse vectors (`dict`s)
- [ ] argument checks
- [ ] lp-norms & metrics
- [ ] `vechadamardminmax`
- [ ] never use `numpy.int64`, they don't detect overflows
- [ ] C++ & Java version
- [ ] Ballin
- [ ] Fields medal

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
