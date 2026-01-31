"""Dense, immutable vector operations.

```python
>>> from vector import vecadd
>>> a = (5, 6, 7)
>>> b = [2]
>>> c = range(4)
>>> vecadd(a, b, c)
(7, 7, 9, 3)
```

Prefixed by `vec...` (vector).

All functions **accept vectors as single exhaustible iterables**.

They **return vectors as tuples**.

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
