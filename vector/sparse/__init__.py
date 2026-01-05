"""Sparse vectors.

```python
>>> from vector import vecsadd
>>> v = {0:1}
>>> w = {0:2, 3:4}
>>> vecsadd(a, b)
{0:3, 3:4}
```

Prefixed by `vecs...` (vector - sparse).

All functions accept vectors and return them as **`dict`s (index:coefficient)**.

The functions are **type-independent**. However, the data types used must
*support necessary scalar operations*. For instance, for vector addition,
coefficients must be addable.

Index keys are expected to be integers.

## Docstring conventions

Summary

Math notation (vector notation if possible, index notation, domain & codomain)

More information ("More efficient than ...").

Complexity
----------
For a vector with $n$ elements there will be - $x$ scalar additions (`add`), ...

Notes
-----
Design choices

See also
--------
Similar functions

References
----------
Wikipedia, numpy, ...
"""

from .creation import *
from .conversion import *
from .utility import *
from .hilbert_space import *
from .vector_space import *
from .elementwise import *
from .objectoriented import *
