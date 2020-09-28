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
        ix = tuple([randrange(m) for m in np.shape(x)])

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
        (np.rad2deg, lambda x: 1 / pi * 180.0, None),
        (np.deg2rad, lambda x: pi / 180.0, None),
    ],
)
def test_unary_function(backend, mode, func, y_d, domain):
    if domain is None:
        x_arr = generate_test_data()
    else:
        x_arr = generate_test_data(a=domain[0], b=domain[1])
    expect_diff = [y_d(xa) for xa in x_arr]
    try:
        with ua.set_backend(udiff.DiffArrayBackend(backend, mode=mode), coerce=True):
            x = np.asarray(x_arr)
            y = func(x)
            x_diff = y.to(x)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(y, da.Array):
        y.compute()

    assert_allclose(x_diff.value, expect_diff)


@pytest.mark.parametrize(
    "func, u_d, v_d, u_domain, v_domain",
    [
        (np.add, lambda u, v: 1, lambda u, v: 1, None, None),
        (np.subtract, lambda u, v: 1, lambda u, v: -1, None, None),
        (np.multiply, lambda u, v: v, lambda u, v: u, None, None),
        (np.divide, lambda u, v: 1 / v, lambda u, v: -u / v ** 2, None, (0, None)),
        (
            np.maximum,
            lambda u, v: 1 if u >= v else 0,
            lambda u, v: 1 if v > u else 0,
            None,
            None,
        ),
        (
            np.minimum,
            lambda u, v: 1 if u <= v else 0,
            lambda u, v: 1 if v <= u else 0,
            None,
            None,
        ),
        (
            np.logaddexp,
            lambda u, v: exp(u) / (exp(u) + exp(v)),
            lambda u, v: exp(v) / (exp(u) + exp(v)),
            (-1, 1),
            (-1, 1),
        ),
        (
            np.logaddexp2,
            lambda u, v: 2 ** u / (2 ** u + 2 ** v),
            lambda u, v: 2 ** v / (2 ** u + 2 ** v),
            (-1, 1),
            (-1, 1),
        ),
        (
            np.true_divide,
            lambda u, v: 1 / v,
            lambda u, v: -u / (v ** 2),
            (1, 5),
            (1, 5),
        ),
        (np.mod, lambda u, v: 1, lambda u, v: -floor(u / v), (1, 10), (1, 10)),
        (
            np.power,
            lambda u, v: pow(u, v) * v / u,
            lambda u, v: pow(u, v) * log(u),
            (1, 5),
            (1, 5),
        ),
        (
            np.arctan2,
            lambda u, v: v / (u ** 2 + v ** 2),
            lambda u, v: -u / (u ** 2 + v ** 2),
            (0, 1),
            (0, 1),
        ),
        (
            np.hypot,
            lambda u, v: u / sqrt(u ** 2 + v ** 2),
            lambda u, v: v / sqrt(u ** 2 + v ** 2),
            None,
            None,
        ),
    ],
)
def test_binary_function(backend, mode, func, u_d, v_d, u_domain, v_domain):
    if u_domain is None:
        u_arr = generate_test_data()
    else:
        u_arr = generate_test_data(a=u_domain[0], b=u_domain[1])
    if v_domain is None:
        v_arr = generate_test_data()
    else:
        v_arr = generate_test_data(a=v_domain[0], b=v_domain[1])

    expect_u_diff = [u_d(ua, va) for ua, va in zip(u_arr, v_arr)]
    expect_v_diff = [v_d(ua, va) for ua, va in zip(u_arr, v_arr)]
    try:
        with ua.set_backend(udiff.DiffArrayBackend(backend, mode=mode), coerce=True):
            u = np.asarray(u_arr)
            v = np.asarray(v_arr)
            y = func(u, v)
            u_diff = y.to(u)
            v_diff = y.to(v)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")
    except NotImplementedError:
        pytest.xfail(
            reason="The func has no implementation in the {} mode.".format(mode)
        )

    if isinstance(y, da.Array):
        y.compute()

    assert_allclose(u_diff.value, expect_u_diff)
    assert_allclose(v_diff.value, expect_v_diff)


@pytest.mark.parametrize(
    "func, y_d, domain",
    [
        (lambda x: x * x, lambda x: 2 * x, None),
        (lambda x: (2 * x + 1) ** 3, lambda x: 6 * (2 * x + 1) ** 2, (0.5, None)),
        (
            lambda x: np.sin(x ** 2) / np.sin(x) ** 2,
            lambda x: (2 * x * cos(x ** 2) * sin(x) - 2 * sin(x ** 2) * cos(x))
            / sin(x) ** 3,
            (0, pi),
        ),
        (
            lambda x: np.log(x ** 2) ** (1 / 3),
            lambda x: 2 * log(x ** 2) ** (-2 / 3) / (3 * x),
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
                (2 * x * atanh(x) / (1 + x ** 2)) - (log(1 + x ** 2) / (1 - x ** 2))
            )
            / atanh(x) ** 2,
            (0, 1),
        ),
    ],
)
def test_arbitrary_function(backend, mode, func, y_d, domain):
    if domain is None:
        x_arr = generate_test_data()
    else:
        x_arr = generate_test_data(a=domain[0], b=domain[1])
    expect_diff = [y_d(xa) for xa in x_arr]
    try:
        with ua.set_backend(udiff.DiffArrayBackend(backend, mode=mode), coerce=True):
            x = np.asarray(x_arr)
            y = func(x)
            x_diff = y.to(x)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(y, da.Array):
        y.compute()

    assert_allclose(x_diff.value, expect_diff)


@pytest.mark.parametrize(
    "func, y_d, domain",
    [
        (lambda x: x * x, lambda x: 2, None),
        (lambda x: (2 * x + 1) ** 3, lambda x: 24 * (2 * x + 1), (0.5, None)),
        (
            lambda x: np.sin(x ** 2) + np.sin(x) ** 2,
            lambda x: 2 * cos(x ** 2)
            - 4 * x ** 2 * sin(x ** 2)
            + 2 * cos(x) ** 2
            - 2 * sin(x) ** 2,
            (0, pi),
        ),
        (
            lambda x: np.log(x ** 2),
            lambda x: -2 / x ** 2,
            (1, None),
        ),
        (
            lambda x: np.power(np.cos(x), 2) * np.log(x),
            lambda x: -2 * cos(2 * x) * log(x)
            - 2 * sin(2 * x) / x
            - cos(x) ** 2 / x ** 2,
            (0, None),
        ),
        (
            lambda x: x / np.sqrt(1 - x ** 2),
            lambda x: 3 * x / (1 - x ** 2) ** (5 / 2),
            (-1, 1),
        ),
    ],
)
def test_high_order_diff(backend, mode, func, y_d, domain):
    if domain is None:
        x_arr = generate_test_data()
    else:
        x_arr = generate_test_data(a=domain[0], b=domain[1])
    expect_diff = [y_d(xa) for xa in x_arr]
    try:
        with ua.set_backend(udiff.DiffArrayBackend(backend, mode=mode), coerce=True):
            x = np.asarray(x_arr)
            y = func(x)
            x_diff = y.to(x).to(x)
    except ua.BackendNotImplementedError:
        if backend in FULLY_TESTED_BACKENDS:
            raise
        pytest.xfail(reason="The backend has no implementation for this ufunc.")

    if isinstance(y, da.Array):
        y.compute()

    assert_allclose(x_diff.value, expect_diff)
