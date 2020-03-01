import uarray as ua
import unumpy as np
import numpy as onp
import torch
import dask.array as da
import udiff
import sparse
from math import *
import unumpy.numpy_backend as NumpyBackend

import unumpy.torch_backend as TorchBackend
import unumpy.dask_backend as DaskBackend
import unumpy.sparse_backend as SparseBackend

from numpy.testing import *

import pytest

ua.set_global_backend(NumpyBackend)

LIST_BACKENDS = [
    NumpyBackend,
    # DaskBackend,
    # SparseBackend,
    pytest.param(
        TorchBackend,
        marks=pytest.mark.xfail(reason="PyTorch not fully NumPy compatible."),
    ),
]


FULLY_TESTED_BACKENDS = [NumpyBackend, DaskBackend]

try:
    import unumpy.xnd_backend as XndBackend
    import xnd
    from ndtypes import ndt

    LIST_BACKENDS.append(XndBackend)
    FULLY_TESTED_BACKENDS.append(XndBackend)
except ImportError:
    XndBackend = None  # type: ignore
    LIST_BACKENDS.append(
        pytest.param(
            None, marks=pytest.mark.skip(reason="xnd is not importable")
        )
    )

try:
    import unumpy.cupy_backend as CupyBackend
    import cupy as cp

    LIST_BACKENDS.append(pytest.param(CupyBackend))
except ImportError:
    LIST_BACKENDS.append(
        pytest.param(
            (None, None), marks=pytest.mark.skip(reason="cupy is not importable")
        )
    )


EXCEPTIONS = {
    (DaskBackend, np.in1d),
    (DaskBackend, np.intersect1d),
    (DaskBackend, np.setdiff1d),
    (DaskBackend, np.setxor1d),
    (DaskBackend, np.union1d),
    (DaskBackend, np.sort),
    (DaskBackend, np.argsort),
    (DaskBackend, np.lexsort),
    (DaskBackend, np.partition),
    (DaskBackend, np.argpartition),
    (DaskBackend, np.sort_complex),
    (DaskBackend, np.msort),
    (DaskBackend, np.searchsorted),
}


@pytest.fixture(scope="session", params=LIST_BACKENDS)
def backend(request):
    backend = request.param
    return backend

x = np.reshape(np.arange(25), (5, 5))

@pytest.mark.parametrize(
    "method, y_d",
    [
        (np.positive, lambda x: 1),
        (np.negative, lambda x: -1),
        (np.exp, lambda x: pow(e, x)),
        (np.exp2, lambda x: pow(2, x) * log(2)),
        (np.log, lambda x: 1 / x),
        (np.log2, lambda x: 1 / (x * log(2))),
        (np.log10, lambda x: 1 / (x * log(10))),
        (np.sqrt, lambda x: 0.5 * pow(x, -0.5)),
        (np.square, lambda x: 2 * x),
        (np.cbrt, lambda x: 1 / 3 * pow(x, -2 / 3)),
        (np.reciprocal, lambda x: -1 / pow(x, 2)),
        (np.sin, lambda x: cos(x)),
        (np.cos, lambda x: -sin(x)),
        (np.tan, lambda x: 1 / cos(x) ** 2),
        (np.arcsin, lambda x: 1 / sqrt(1 - x ** 2)),
        (np.arccos, lambda x: -1 / sqrt(1 - x ** 2)),
        (np.arctan, lambda x: 1 / (1 + x ** 2)),
        (np.arctanh, lambda x: 1 / (1 - x ** 2)),
        (np.sinh, lambda x: cosh(x)),
        (np.cosh, lambda x: sinh(x)),
        (np.tanh, lambda x: 1 / cosh(x) ** 2),
        (np.arcsinh, lambda x: 1 / sqrt(1 + x ** 2)),
        (np.arccosh, lambda x: 1 / sqrt(-1 + x ** 2 )),
        (np.arctanh, lambda x: 1 / (1 - x ** 2))
    ],
)
def test_unary_function(backend, method, y_d):
    x_arr = [0.2, 0.3]
    y_d_arr = [y_d(xr) for xr in x_arr]
    try:
        with ua.set_backend(backend), ua.set_backend(udiff, coerce=True):
            x = np.asarray(x_arr)
            x.var = udiff.Variable('x')
            ret = method(x)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()

    assert_allclose(ret.diffs[x].arr, y_d_arr)

@pytest.mark.parametrize(
    "func, y_d",
    [
        (lambda x: np.power(2 * x + 1, 3), lambda x: 6 * np.power(2 * x + 1, 2)),
        (lambda x: np.sin(np.power(x, 2)) / np.power(np.sin(x), 2), lambda x: (2 * x * np.cos(np.power(x, 2)) * np.sin(x) - 2 * np.sin(np.power(x, 2)) * np.cos(x)) / np.power(np.sin(x), 3)),
        (lambda x: np.power(np.log(np.power(x, 3)), 1/3), lambda x: 2 * np.power(np.log(np.power(x, 2)), -2/3) / (3 * x)),
        (lambda x: np.log((1 + x) / (1 - x)) / 4 - np.arctan(x) / 2, lambda x: np.power(x, 2) / (1 - np.power(x, 4))),
        (lambda x: np.arctanh(3 * x ** 3 + x ** 2 +1), lambda x: (9 * x ** 2 + 2 * x) / (1 - np.power(3 * x ** 3 + x ** 2 + 1 , 2))),
        (lambda x: np.sinh(np.cbrt(x)) + np.cosh(4 * x ** 3) , lambda x: np.cosh(np.cbrt(x)) / (3 * x ** (2/3)) + 12 * (x ** 2) * np.sinh(4 * x ** 3)),
        (lambda x: np.log(1 + x ** 2) / np.arctanh(x), lambda x: ((2 * x * np.arctanh(x) / (1 + x ** 2)) - (np.log(1 + x ** 2)/(1 - x ** 2))) / np.power(np.arctanh(x) , 2))
    ],
)
def test_arbitrary_function(backend, func, y_d):
    x_arr = [0.2, 0.3]
    try:
        with ua.set_backend(backend), ua.set_backend(udiff, coerce=True):
            x = np.asarray(x_arr)
            x.var = udiff.Variable('x')
            ret = func(x)
            y_d_arr = y_d(x)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()

    assert_allclose(ret.diffs[x].arr, y_d_arr.arr)
