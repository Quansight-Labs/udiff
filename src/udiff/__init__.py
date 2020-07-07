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
>>> import numpy as onp
>>> with ua.set_backend(numpy_backend), ua.set_backend(udiff, coerce=True):
...    x1 = np.array([2])
...    x2 = np.array([5])
...    y = np.log(x1) + x1 * x2 - np.sin(x2)
...    y.backward()
...    print(allclose(x1.diff, [5.5]))
True

"""

import sys
import uarray as ua

from . import _vjp_diffs
from ._uarray_plug import __ua_domain__, __ua_convert__, __ua_function__
from ._vjp_core import defvjp

from ._diff_array import DiffArray

__all__ = [
    "__ua_domain__",
    "__ua_convert__",
    "__ua_function__",
    "defvjp",
    "DiffArray",
]

SKIP_SELF = ua.skip_backend(sys.modules["udiff"])
