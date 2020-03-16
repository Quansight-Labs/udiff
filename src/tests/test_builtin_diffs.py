import uarray as ua
import unumpy as np
import numpy as onp
import torch
import dask.array as da
import udiff
import sparse
from math import *
from random import uniform
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

def generate_test_data(n_elements=10, a=None, b=None):
    if a is None:
        a = -10
    if b is None:
        b = 10
    x_arr = [uniform(a + 1e-3, b - 1e-3) for i in range(n_elements)]
    return x_arr

@pytest.mark.parametrize(
    "method, y_d, domain",
    [
        (np.positive, lambda x: 1, None),
        (np.negative, lambda x: -1, None),
        (np.exp, lambda x: pow(e, x), None),
        (np.exp2, lambda x: pow(2, x) * log(2), None),
        (np.log, lambda x: 1 / x, (0, None)),
        (np.log2, lambda x: 1 / (x * log(2)), (0, None)),
        (np.log10, lambda x: 1 / (x * log(10)), (0, None)),
        (np.sqrt, lambda x: 0.5 * pow(x, -0.5), (0, None)),
        (np.square, lambda x: 2 * x, None),
        (np.cbrt, lambda x: 1 / 3 * pow(x, -2 / 3), (0, None)), # Negative numbers cannot be raised to a fractional power
        (np.reciprocal, lambda x: -1 / pow(x, 2), (None, 0)),
        (np.sin, lambda x: cos(x), None),
        (np.cos, lambda x: -sin(x), None),
        (np.tan, lambda x: 1 / cos(x) ** 2, None),
        (np.arcsin, lambda x: 1 / sqrt(1 - x ** 2), (-1, 1)),
        (np.arccos, lambda x: -1 / sqrt(1 - x ** 2), (-1, 1)),
        (np.arctan, lambda x: 1 / (1 + x ** 2), None),
        (np.sinh, lambda x: cosh(x), None),
        (np.cosh, lambda x: sinh(x), (1, None)),
        (np.tanh, lambda x: 1 / cosh(x) ** 2, (-1, 1)),
        (np.arcsinh, lambda x: 1 / sqrt(1 + x ** 2), None),
        (np.arccosh, lambda x: 1 / sqrt(-1 + x ** 2), (1, None)),
        (np.arctanh, lambda x: 1 / (1 - x ** 2), (-1, 1))
    ],
)
def test_unary_function(backend, method, y_d, domain):
    if domain is None:
        x_arr = generate_test_data()
    else:
        x_arr = generate_test_data(a=domain[0], b=domain[1])
    y_d_arr = [y_d(xa) for xa in x_arr]
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
    "func, y_d, domain",
    [
        (lambda x: (2 * x + 1) ** 3, lambda x: 6 * (2 * x + 1) ** 2, (0.5, None)),
        (lambda x: np.sin(x ** 2) / (np.sin(x)) ** 2, lambda x: (2 * x * np.cos(x ** 2) * np.sin(x) - 2 * np.sin(x ** 2) * np.cos(x)) / (np.sin(x)) ** 3, (0, pi)),
        (lambda x: (np.log(x ** 2)) ** (1 / 3), lambda x: 2 * (np.log(x ** 2)) ** (-2/3) / (3 * x), (1, None)),
        (lambda x: np.log((1 + x) / (1 - x)) / 4 - np.arctan(x) / 2, lambda x: x ** 2 / (1 - x ** 4), (-1, 1)),
        (lambda x: np.arctanh(3 * x ** 3 + x ** 2 + 1), lambda x: (9 * x ** 2 + 2 * x) / (1 - (3 * x ** 3 + x ** 2 + 1) ** 2), (0, None)),
        (lambda x: np.sinh(np.cbrt(x)) + np.cosh(4 * x ** 3) , lambda x: np.cosh(np.cbrt(x)) / (3 * x ** (2/3)) + 12 * (x ** 2) * np.sinh(4 * x ** 3), (1/4, None)),
        (lambda x: np.log(1 + x ** 2) / np.arctanh(x), lambda x: ((2 * x * np.arctanh(x) / (1 + x ** 2)) - (np.log(1 + x ** 2) / (1 - x ** 2))) / (np.arctanh(x)) ** 2, (0, 1))
    ],
)
def test_arbitrary_function(backend, func, y_d, domain):
    if domain is None:
        x_arr = generate_test_data()
    else:
        x_arr = generate_test_data(a=domain[0], b=domain[1])
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
