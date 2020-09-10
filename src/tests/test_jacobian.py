import uarray as ua
import unumpy as np
import numpy as onp
import torch
import dask.array as da
import udiff
import sparse
from math import *
from random import uniform, randrange
import unumpy.numpy_backend as NumpyBackend

import unumpy.torch_backend as TorchBackend
import unumpy.dask_backend as DaskBackend
import unumpy.sparse_backend as SparseBackend

import numpy as onp
from numpy.testing import assert_allclose

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


@pytest.fixture(scope="session", params=["vjp", "jvp"])
def mode(request):
    mode = request.param
    return mode


@pytest.mark.parametrize(
    "x, func, expect_jacobian",
    [
        (
            onp.arange(12).reshape(2, 3, 2),
            lambda x: np.sum(x, axis=1),
            [
                [
                    [[[1, 0], [1, 0], [1, 0]], [[0, 0], [0, 0], [0, 0]]],
                    [[[0, 1], [0, 1], [0, 1]], [[0, 0], [0, 0], [0, 0]]],
                ],
                [
                    [[[0, 0], [0, 0], [0, 0]], [[1, 0], [1, 0], [1, 0]]],
                    [[[0, 0], [0, 0], [0, 0]], [[0, 1], [0, 1], [0, 1]]],
                ],
            ],
        ),
        (
            onp.arange(4).reshape((2, 2)),
            lambda x: x,
            [
                [[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                [[[0, 0], [1, 0]], [[0, 0], [0, 1]]],
            ],
        ),
    ],
)
def test_jacobian(backend, mode, x, func, expect_jacobian):
    try:
        with ua.set_backend(udiff.DiffArrayBackend(backend, mode=mode), coerce=True):
            x = np.asarray(x)
            y = func(x)
            x_jacobian = y.to(x, jacobian=True)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(y, da.Array):
        y.compute()

    assert_allclose(x_jacobian.value, expect_jacobian)


@pytest.mark.parametrize(
    "u, v, func, expect_u_jacobian, expect_v_jacobian",
    [
        (
            onp.arange(2).reshape(1, 2, 1),
            onp.arange(2).reshape(1, 1, 2),
            lambda x, y: np.matmul(x, y),
            [[[[[[0], [0]]], [[[1], [0]]]], [[[[0], [0]]], [[[0], [1]]]]]],
            [[[[[[0, 0]]], [[[0, 0]]]], [[[[1, 0]]], [[[0, 1]]]]]],
        ),
    ],
)
def test_separation_binary(
    backend, mode, u, v, func, expect_u_jacobian, expect_v_jacobian
):
    try:
        with ua.set_backend(udiff.DiffArrayBackend(backend, mode=mode), coerce=True):
            u = np.asarray(u)
            v = np.asarray(v)

            y = func(u, v)
            u_jacobian = y.to(u, jacobian=True)
            v_jacobian = y.to(v, jacobian=True)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(y, da.Array):
        y.compute()

    assert_allclose(u_jacobian.value, expect_u_jacobian)
    assert_allclose(v_jacobian.value, expect_v_jacobian)
