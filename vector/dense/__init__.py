"""Dense vector operations on sequences.

```python
>>> from vector import vecadd
>>> a = (5, 6, 7)
>>> b = [2]
>>> c = range(4)
>>> vecadd(a, b, c)
(7, 7, 9, 3)
```

Two families of functions:

- **`vec...`** — accept any iterable, return a new sequence
  (default `tuple`, configurable via `factory`).
- **`veci...`** — accept a `MutableSequence`, mutate it in-place and return it.

The functions are **type-independent**. However, the coefficients used must
**support necessary scalar operations**. For instance, for vector addition,
coefficients must be addable.

For complete type safety a **zero** argument is available. Default is `int(0)`.

## Docstring conventions

Summary

Math notation (vector notation if possible, index notation, domain & codomain)

More information ("More efficient than ...").

Complexity
----------
For a vector of length $n$ there will be - $x$ scalar additions (`add`), ...

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
from .utility import *
from .hilbertspace import *
from .vectorspace import *
from .elementwise import *
from .objectoriented import *
