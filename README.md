# `udiff` - Automatic differentiation with uarray/unumpy.
[![Join the chat at https://gitter.im/Plures/uarray](https://badges.gitter.im/Plures/uarray.svg)](https://gitter.im/Plures/uarray?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) ![language](https://img.shields.io/badge/language-python3-orange.svg) ![license](https://img.shields.io/github/license/Quansight-Labs/udiff)

## Quickstart
```python
import uarray as ua
import unumpy as np
import udiff
from unumpy import numpy_backend

with ua.set_backend(udiff.DiffArrayBackend(numpy_backend), coerce=True):
    x1 = np.reshape(np.arange(1, 26), (5, 5))
    x2 = np.reshape(np.arange(1, 26), (5, 5))
    y = np.log(x1) + x1 * x2 - np.sin(x2)
    print(y)
    print(y.to(x1))
    print(y.to(x2))
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more information on how to contribute to `udiff`.
