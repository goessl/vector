"""Vector in-place operations.

```python
>>> from vector import vecimul
>>> v = [1, 2, 3]
>>> vecimul(2, v)
[2, 4, 6]
>>> v
[2, 4, 6]
```

Prefixed by `veci...` (vector - in-place).

All functions **accept vectors as lists**.

They modify them **in-place**.

The functions are **type-independent**. However, the coefficients used must
**support necessary scalar operations**. For instance, for vector addition,
coefficients must be addable.

For complete type safety a **zero** argument is available. Default is `int(0)`.

## Docstring conventions

Summary

Math notation (vector notation if possible, index notation, domain & codomain)

More information ("More efficient than ...").

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
