import unumpy as np
from ._vjp_diffs import (
    balanced_eq,
    match_complex,
    replace_zero,
    metadata,
    # dot_adjoint_0,
    # dot_adjoint_1,
    # tensordot_adjoint_0,
    # tensordot_adjoint_1,
    nograd_functions,
)
from ._core import (
    defjvp,
    defjvp_argnum,
    def_linear,
)

# ----- Functions that are constant w.r.t. continuous inputs -----
defjvp(np.nan_to_num, lambda g, ans, x: np.where(np.isfinite(x), g, 0.0))

# ----- Binary ufuncs (linear) -----
def_linear(np.multiply)

# ----- Binary ufuncs -----
defjvp(
    np.add,
    lambda g, ans, x, y: broadcast(g, ans),
    lambda g, ans, x, y: broadcast(g, ans),
)
defjvp(
    np.subtract,
    lambda g, ans, x, y: broadcast(g, ans),
    lambda g, ans, x, y: broadcast(-g, ans),
)
defjvp(np.divide, "same", lambda g, ans, x, y: -g * x / y ** 2)
defjvp(
    np.maximum,
    lambda g, ans, x, y: g * balanced_eq(x, ans, y),
    lambda g, ans, x, y: g * balanced_eq(y, ans, x),
)
defjvp(
    np.minimum,
    lambda g, ans, x, y: g * balanced_eq(x, ans, y),
    lambda g, ans, x, y: g * balanced_eq(y, ans, x),
)
defjvp(
    np.fmax,
    lambda g, ans, x, y: g * balanced_eq(x, ans, y),
    lambda g, ans, x, y: g * balanced_eq(y, ans, x),
)
defjvp(
    np.fmin,
    lambda g, ans, x, y: g * balanced_eq(x, ans, y),
    lambda g, ans, x, y: g * balanced_eq(y, ans, x),
)
defjvp(
    np.logaddexp,
    lambda g, ans, x, y: g * np.exp(x - ans),
    lambda g, ans, x, y: g * np.exp(y - ans),
)
defjvp(
    np.logaddexp2,
    lambda g, ans, x, y: g * 2 ** (x - ans),
    lambda g, ans, x, y: g * 2 ** (y - ans),
)
defjvp(np.true_divide, "same", lambda g, ans, x, y: -g * x / y ** 2)
defjvp(
    np.mod,
    lambda g, ans, x, y: broadcast(g, ans),
    lambda g, ans, x, y: -g * np.floor(x / y),
)
defjvp(
    np.remainder,
    lambda g, ans, x, y: broadcast(g, ans),
    lambda g, ans, x, y: -g * np.floor(x / y),
)
defjvp(
    np.power,
    lambda g, ans, x, y: g * y * x ** np.where(y, y - 1, 1.0),
    lambda g, ans, x, y: g * np.log(replace_zero(x, 1.0)) * ans,
)
defjvp(
    np.arctan2,
    lambda g, ans, x, y: g * y / (x ** 2 + y ** 2),
    lambda g, ans, x, y: g * -x / (x ** 2 + y ** 2),
)

# ----- Simple grads (linear) -----
defjvp(np.negative, "same")
defjvp(np.rad2deg, "same")
defjvp(np.degrees, "same")
defjvp(np.deg2rad, "same")
defjvp(np.radians, "same")
defjvp(np.reshape, "same")
defjvp(np.roll, "same")
defjvp(np.array_split, "same")
defjvp(np.split, "same")
defjvp(np.vsplit, "same")
defjvp(np.hsplit, "same")
defjvp(np.dsplit, "same")
defjvp(np.ravel, "same")
defjvp(np.expand_dims, "same")
defjvp(np.squeeze, "same")
defjvp(np.diag, "same")
defjvp(np.diagonal, "same")
# defjvp(np.make_diagonal, "same")
defjvp(np.flipud, "same")
defjvp(np.fliplr, "same")
defjvp(np.rot90, "same")
# defjvp(np.trace, "same")
defjvp(np.full, "same", argnums=(1,))
defjvp(np.triu, "same")
defjvp(np.tril, "same")
defjvp(np.swapaxes, "same")
defjvp(np.rollaxis, "same")
defjvp(np.moveaxis, "same")
defjvp(np.broadcast_to, "same")
def_linear(np.cross)

# ----- Simple grads -----
# defjvp(
#     np.abs,
#     lambda g, ans, x: np.real(g * replace_zero(np.conj(x), 0.0))
#     / replace_zero(ans, 1.0),
# )
defjvp(np.fabs, lambda g, ans, x: np.sign(x) * g)  # fabs doesn't take complex numbers.
defjvp(np.absolute, lambda g, ans, x: np.real(g * np.conj(x)) / ans)
defjvp(np.reciprocal, lambda g, ans, x: -g / x ** 2)
defjvp(np.exp, lambda g, ans, x: ans * g)
defjvp(np.exp2, lambda g, ans, x: ans * np.log(2) * g)
defjvp(np.expm1, lambda g, ans, x: (ans + 1) * g)
defjvp(np.log, lambda g, ans, x: g / x)
defjvp(np.log2, lambda g, ans, x: g / x / np.log(2))
defjvp(np.log10, lambda g, ans, x: g / x / np.log(10))
defjvp(np.log1p, lambda g, ans, x: g / (x + 1))
defjvp(np.sin, lambda g, ans, x: g * np.cos(x))
defjvp(np.cos, lambda g, ans, x: -g * np.sin(x))
defjvp(np.tan, lambda g, ans, x: g / np.cos(x) ** 2)
defjvp(np.arcsin, lambda g, ans, x: g / np.sqrt(1 - x ** 2))
defjvp(np.arccos, lambda g, ans, x: -g / np.sqrt(1 - x ** 2))
defjvp(np.arctan, lambda g, ans, x: g / (1 + x ** 2))
defjvp(np.sinh, lambda g, ans, x: g * np.cosh(x))
defjvp(np.cosh, lambda g, ans, x: g * np.sinh(x))
defjvp(np.tanh, lambda g, ans, x: g / np.cosh(x) ** 2)
defjvp(np.arcsinh, lambda g, ans, x: g / np.sqrt(x ** 2 + 1))
defjvp(np.arccosh, lambda g, ans, x: g / np.sqrt(x ** 2 - 1))
defjvp(np.arctanh, lambda g, ans, x: g / (1 - x ** 2))
defjvp(np.square, lambda g, ans, x: g * 2 * x)
defjvp(np.sqrt, lambda g, ans, x: g * 0.5 * x ** -0.5)
defjvp(
    np.sinc,
    lambda g, ans, x: g
    * (np.cos(np.pi * x) * np.pi * x - np.sin(np.pi * x))
    / (np.pi * x ** 2),
)
defjvp(
    np.clip,
    lambda g, ans, x, a_min, a_max: g * np.logical_and(ans != a_min, ans != a_max),
)
defjvp(np.real_if_close, lambda g, ans, x: match_complex(ans, g))
defjvp(np.real, lambda g, ans, x: np.real(g))
defjvp(np.imag, lambda g, ans, x: match_complex(ans, -1j * g))
defjvp(np.conj, lambda g, ans, x: np.conj(g))
defjvp(
    np.angle,
    lambda g, ans, x: match_complex(ans, g * np.conj(x * 1j) / np.abs(x) ** 2),
)
defjvp(
    np.where,
    None,
    lambda g, ans, c, x=None, y=None: np.where(c, g, np.zeros(np.shape(g))),
    lambda g, ans, c, x=None, y=None: np.where(c, np.zeros(g.shape), g),
)

# ----- Trickier grads -----
# defjvp(np.kron, "same", "same")
defjvp(np.diff, "same")
defjvp(np.gradient, "same")
defjvp(np.repeat, "same")
defjvp(np.tile, "same")
defjvp(np.transpose, "same")
defjvp(np.sum, "same")
# defjvp(np.mean, "same")
defjvp(
    np.prod,
    lambda g, ans, x, axis=None, keepdims=False: ans
    * np.sum(g / x, axis=axis, keepdims=keepdims),
)
defjvp(
    np.linspace,
    lambda g, ans, start, stop, *args, **kwargs: np.linspace(g, 0, *args, **kwargs),
    lambda g, ans, start, stop, *args, **kwargs: np.linspace(0, g, *args, **kwargs),
)


def forward_grad_np_var(g, ans, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        num_reps = np.size(g)
    elif isinstance(axis, int):
        num_reps = np.shape(g)[axis]
    elif isinstance(axis, tuple):
        num_reps = np.prod(np.array(np.shape(g))[list(axis)])

    x_minus_mean = np.conj(x - np.mean(x, axis=axis, keepdims=True))
    return (
        2.0
        * np.sum(np.real(g * x_minus_mean), axis=axis, keepdims=keepdims)
        / (num_reps - ddof)
    )


defjvp(np.var, forward_grad_np_var)


def forward_grad_np_std(g, ans, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        num_reps = np.size(g)
    elif isinstance(axis, int):
        num_reps = np.shape(g)[axis]
    elif isinstance(axis, tuple):
        num_reps = np.prod(np.array(np.shape(g))[list(axis)])

    if num_reps <= 1:
        return np.zeros_like(ans)
    x_minus_mean = np.conj(x - np.mean(x, axis=axis, keepdims=True))
    return np.sum(np.real(g * x_minus_mean), axis=axis, keepdims=keepdims) / (
        (num_reps - ddof) * ans
    )


defjvp(np.std, forward_grad_np_std)


def fwd_grad_chooser(g, ans, x, axis=None, keepdims=False):
    if np.isscalar(x):
        return g
    if not keepdims:
        if isinstance(axis, int):
            ans = np.expand_dims(ans, axis)
        elif isinstance(axis, tuple):
            for ax in sorted(axis):
                ans = np.expand_dims(ans, ax)
    chosen_locations = x == ans
    return np.sum((g * chosen_locations), axis=axis, keepdims=keepdims) / np.sum(
        chosen_locations, axis=axis, keepdims=keepdims
    )


defjvp(np.max, fwd_grad_chooser)
defjvp(np.min, fwd_grad_chooser)
# defjvp(np.amax, fwd_grad_chooser)
# defjvp(np.amin, fwd_grad_chooser)

defjvp(np.cumsum, "same")

# def_linear(np.inner)
def_linear(np.matmul)
# def_linear(np.dot)
# def_linear(np.tensordot)
# def_linear(np.outer)

# def_linear(dot_adjoint_0)
# def_linear(dot_adjoint_1)

# def_linear(tensordot_adjoint_0)
# def_linear(tensordot_adjoint_1)


def fwd_grad_sort(g, ans, x, axis=-1, kind="quicksort", order=None):
    sort_perm = np.argsort(x, axis, kind, order)
    return g[sort_perm]


defjvp(np.sort, fwd_grad_sort)
defjvp(np.msort, lambda g, ans, x: fwd_grad_sort(g, ans, x, axis=0))


def fwd_grad_partition(g, ans, x, kth, axis=-1, kind="introselect", order=None):
    partition_perm = np.argpartition(x, kth, axis, kind, order)
    return g[partition_perm]


defjvp(np.partition, fwd_grad_partition)


def atleast_jvpmaker(fun):
    def jvp(g, ans, *arys):
        if len(arys) > 1:
            raise NotImplementedError("Can't handle multiple arguments yet.")
        return fun(g)

    return jvp


defjvp(np.atleast_1d, atleast_jvpmaker(np.atleast_1d))
defjvp(np.atleast_2d, atleast_jvpmaker(np.atleast_2d))
defjvp(np.atleast_3d, atleast_jvpmaker(np.atleast_3d))

# def_linear(np.einsum)


def broadcast(x, target):
    target_shape, target_ndim, target_dtype, target_iscomplex = metadata(target)
    while np.ndim(x) < target_ndim:
        x = np.expand_dims(x, 0)
    for axis, size in enumerate(np.shape(x)):
        if size == 1:
            x = np.repeat(x, target_shape[axis], axis=axis)
        # x = x + 0j
    return x


defjvp(np.pad, lambda g, ans, array, width, mode, **kwargs: np.pad(g, width, mode))
