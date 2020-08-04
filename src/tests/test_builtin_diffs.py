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

# try:
#     import unumpy.xnd_backend as XndBackend
#     import xnd
#     from ndtypes import ndt

#     LIST_BACKENDS.append(XndBackend)
#     FULLY_TESTED_BACKENDS.append(XndBackend)
# except ImportError:
#     XndBackend = None  # type: ignore
#     LIST_BACKENDS.append(
#         pytest.param(None, marks=pytest.mark.skip(reason="xnd is not importable"))
#     )

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


def generate_test_data(n_elements=12, a=None, b=None):
    if a is None:
        a = -10
    if b is None:
        b = 10
    x_arr = [uniform(a + 1e-3, b - 1e-3) for i in range(n_elements)]
    return x_arr


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in this dimensions.
    """
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = ((fxph - fxmh) / (2 * h))[ix]
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(grad_numerical)
        print(grad_analytic)
        print(rel_error)
        assert_almost_equal(rel_error, 0, decimal=5)


@pytest.mark.parametrize(
    "func, y_d, domain",
    [
        (np.positive, lambda x: 1, None),
        (np.negative, lambda x: -1, None),
        (np.exp, lambda x: pow(e, x), None),
        (np.exp2, lambda x: pow(2, x) * log(2), None),
        (np.log, lambda x: 1 / x, (0, None)),
        (np.log2, lambda x: 1 / (x * log(2)), (0, None)),
        (np.log10, lambda x: 1 / (x * log(10)), (0, None)),
        (np.log1p, lambda x: 1 / (x + 1), (-1, None)),
        (np.sqrt, lambda x: 0.5 * pow(x, -0.5), (0, None)),
        (np.square, lambda x: 2 * x, None),
        (np.reciprocal, lambda x: -1 / pow(x, 2), (None, 0)),
        (np.sin, lambda x: cos(x), None),
        (np.cos, lambda x: -sin(x), None),
        (
            np.tan,
            lambda x: 1 / cos(x) ** 2,
            (-5, 5),
        ),  # Set bound to prevent numerical overflow
        (np.arcsin, lambda x: 1 / sqrt(1 - x ** 2), (-1, 1)),
        (np.arccos, lambda x: -1 / sqrt(1 - x ** 2), (-1, 1)),
        (np.arctan, lambda x: 1 / (1 + x ** 2), None),
        (np.sinh, lambda x: cosh(x), None),
        (np.cosh, lambda x: sinh(x), (1, None)),
        (np.tanh, lambda x: 1 / cosh(x) ** 2, (-1, 1)),
        (np.arcsinh, lambda x: 1 / sqrt(1 + x ** 2), None),
        (np.arccosh, lambda x: 1 / sqrt(-1 + x ** 2), (1, None)),
        (np.arctanh, lambda x: 1 / (1 - x ** 2), (-1, 1)),
        (np.absolute, lambda x: 1 if x > 0 else -1, None),
        (np.fabs, lambda x: 1 if x > 0 else -1, None),
        (np.reciprocal, lambda x: -1 / x ** 2, (1, 10)),
        (np.expm1, lambda x: exp(x), None),
        # (np.rad2deg, lambda x: 1 / pi * 180.0, None),
        # (np.deg2rad, lambda x: pi / 180.0, None),
    ],
)
def test_unary_function(backend, func, y_d, domain):
    if domain is None:
        x_arr = generate_test_data()
    else:
        x_arr = generate_test_data(a=domain[0], b=domain[1])
    y_d_arr = [y_d(xa) for xa in x_arr]
    try:
        with ua.set_backend(backend, coerce=True), ua.set_backend(udiff, coerce=True):
            x = np.asarray(x_arr)
            ret = func(x)
            ret.backward()
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()

    assert_allclose(x.diff[ret.name].value, y_d_arr)


@pytest.mark.parametrize(
    "func, y_d, domain",
    [
        (lambda x: x * x, lambda x: 2 * x, None),
        (lambda x: (2 * x + 1) ** 3, lambda x: 6 * (2 * x + 1) ** 2, (0.5, None)),
        (
            lambda x: np.sin(x ** 2) / (np.sin(x)) ** 2,
            lambda x: (
                2 * x * onp.cos(x ** 2) * onp.sin(x) - 2 * onp.sin(x ** 2) * onp.cos(x)
            )
            / (onp.sin(x)) ** 3,
            (0, pi),
        ),
        (
            lambda x: (np.log(x ** 2)) ** (1 / 3),
            lambda x: 2 * (onp.log(x ** 2)) ** (-2 / 3) / (3 * x),
            (1, None),
        ),
        (
            lambda x: np.log((1 + x) / (1 - x)) / 4 - np.arctan(x) / 2,
            lambda x: x ** 2 / (1 - x ** 4),
            (-1, 1),
        ),
        (
            lambda x: np.log(1 + x ** 2) / np.arctanh(x),
            lambda x: (
                (2 * x * onp.arctanh(x) / (1 + x ** 2))
                - (onp.log(1 + x ** 2) / (1 - x ** 2))
            )
            / (onp.arctanh(x)) ** 2,
            (0, 1),
        ),
    ],
)
def test_arbitrary_function(backend, func, y_d, domain):
    if domain is None:
        x_arr = generate_test_data()
    else:
        x_arr = generate_test_data(a=domain[0], b=domain[1])
    y_d_arr = [y_d(xa) for xa in x_arr]
    try:
        with ua.set_backend(backend, coerce=True), ua.set_backend(udiff, coerce=True):
            x = np.asarray(x_arr)
            ret = func(x)
            ret.backward()
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()

    assert_allclose(x.diff[ret.name].value, y_d_arr)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "u, v, func, u_jacobian, v_jacobian",
    [
        (
            onp.arange(2).reshape(1, 2, 1),
            onp.arange(2).reshape(1, 1, 2),
            lambda x, y: np.matmul(x, y),
            [[[[[[0], [0]]], [[[1], [0]]]], [[[[0], [0]]], [[[0], [1]]]]]],
            [[[[[[0, 0]]], [[[0, 0]]]], [[[[1, 0]]], [[[0, 1]]]]]],
        )
    ],
)
def test_separation_binary(backend, u, v, func, u_jacobian, v_jacobian):
    try:
        with ua.set_backend(backend, coerce=True), ua.set_backend(udiff, coerce=True):
            u = np.asarray(u)
            v = np.asarray(v)

            ret = func(u, v)
            u_jacobian = ret.to(u, jacobian=True)
            v_jacobian = ret.to(v, jacobian=True)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()

    assert_allclose(u_jacobian.value, u_jacobian)
    assert_allclose(v_jacobian.value, v_jacobian)


@pytest.mark.parametrize(
    "func, y_d, domain",
    [
        (lambda x: (2 * x + 1) ** 3, lambda x: 6 * (2 * x + 1) ** 2, (0.5, None)),
        (
            lambda x: np.sin(x ** 2) / (np.sin(x)) ** 2,
            lambda x: (
                2 * x * onp.cos(x ** 2) * onp.sin(x) - 2 * onp.sin(x ** 2) * onp.cos(x)
            )
            / (onp.sin(x)) ** 3,
            (0, pi),
        ),
        (
            lambda x: (np.log(x ** 2)) ** (1 / 3),
            lambda x: 2 * (onp.log(x ** 2)) ** (-2 / 3) / (3 * x),
            (1, None),
        ),
        (
            lambda x: np.log((1 + x) / (1 - x)) / 4 - np.arctan(x) / 2,
            lambda x: x ** 2 / (1 - x ** 4),
            (-1, 1),
        ),
        (
            lambda x: np.log(1 + x ** 2) / np.arctanh(x),
            lambda x: (
                (2 * x * onp.arctanh(x) / (1 + x ** 2))
                - (onp.log(1 + x ** 2) / (1 - x ** 2))
            )
            / (onp.arctanh(x)) ** 2,
            (0, 1),
        ),
    ],
)
def test_to_diff(backend, func, y_d, domain):
    if domain is None:
        x_arr = generate_test_data()
    else:
        x_arr = generate_test_data(a=domain[0], b=domain[1])
    y_d_arr = [y_d(xa) for xa in x_arr]
    try:
        with ua.set_backend(backend, coerce=True), ua.set_backend(udiff, coerce=True):
            x = np.asarray(x_arr)
            ret = func(x)
            x_diff = ret.to(x)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()

    assert_allclose(x_diff.value, y_d_arr)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "x, func, x_jacobian",
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
def test_to_jacobian(backend, x, func, x_jacobian):
    try:
        with ua.set_backend(backend, coerce=True), ua.set_backend(udiff, coerce=True):
            x = np.asarray(x)
            ret = func(x)
            compute_jacobian = ret.to(x, jacobian=True)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()

    assert_allclose(compute_jacobian.value, x_jacobian)


@pytest.mark.parametrize(
    "func, y_d, domain",
    [(lambda x: (2 * x + 1) ** 3, lambda x: 24 * (2 * x + 1), (0.5, None)),],
)
def test_high_order_diff(backend, func, y_d, domain):
    if domain is None:
        x_arr = generate_test_data()
    else:
        x_arr = generate_test_data(a=domain[0], b=domain[1])
    y_d_arr = [y_d(xa) for xa in x_arr]
    try:
        with ua.set_backend(backend, coerce=True), ua.set_backend(udiff, coerce=True):
            x = np.asarray(x_arr)
            ret = func(x)
            x_diff = ret.to(x).to(x)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(ret, da.Array):
        ret.compute()

    assert_allclose(x_diff.value, y_d_arr)
