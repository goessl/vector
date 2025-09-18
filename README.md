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

**Enjoy the [documentation webpage](https://goessl.github.io/vector).**

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
