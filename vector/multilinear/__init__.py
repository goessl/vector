"""Tensor arithmetic.

Prefixed by `ten...` (tensor).

Handle multiaxis vectors, that for example represent multivariate polynomials.

Tensors are accepted as [`numpy.array_like`](https://numpy.org/doc/stable/glossary.html#term-array_like) and returned as [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)s.

Broadcasting happens similar to [`numpy`s broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html),
but the axes are matched in ascending order instead of descending order, and
the arrays don't get stretched but rather padded with zeros.
"""



from .creation import *
from .utility import *
from .hilbert_space import *
from .vector_space import *
from .elementwise import *
