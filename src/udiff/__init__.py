"""
Quick Start
-----------

Getting started is easy:

.. note::
    :obj:`udiff` has not been published on PyPI. You have to install it from source code now.

#.  Use Git to clone the :obj:`udiff` repository:

    .. code:: bash

        git clone https://github.com/Quansight-Labs/udiff.git
        cd udiff

#.  Install :obj:`udiff` on the command line, enter:

    .. code:: bash

        python setup.py install --user

#.  Use :obj:`udiff` in your code. A simple example:

>>> import uarray as ua
>>> import unumpy as np
>>> import udiff
>>> from unumpy import numpy_backend
>>> from numpy import allclose
>>> with ua.set_backend(numpy_backend), ua.set_backend(udiff, coerce=True):
...    x = np.reshape(np.arange(25), (5, 5))
...    x.var = udiff.Variable('x')
...    y = np.exp(2 * x)
...    y_d = 2 * y
...    print(allclose(y.diffs[x].arr, y_d.arr))
True

"""

import sys
import uarray as ua


from ._uarray_plug import __ua_domain__, __ua_convert__, __ua_function__
from ._func_diff_registry import (
    FunctionDifferentialRegistry,
    register_diff,
    diff,
)
from ._diff_array import Variable, DiffArray, ArrayDiffRegistry

__all__ = [
    "__ua_domain__",
    "__ua_convert__",
    "__ua_function__",
    "FunctionDifferentialRegistry",
    "register_diff",
    "diff",
    "Variable",
    "DiffArray",
    "ArrayDiffRegistry",
]

SKIP_SELF = ua.skip_backend(sys.modules["udiff"])
