import unumpy as np
from ._vjp_diffs import (
    balanced_eq,
    match_complex,
    replace_zero,
    metadata,
    nograd_functions,
)
from ._core import (
    defjvp,
    defjvp_argnum,
    def_linear,
)

# ----- Functions that are constant w.r.t. continuous inputs -----
defjvp(np.nan_to_num, lambda ans, x: lambda g: np.where(np.isfinite(x), g, 0.0))

# ----- Binary ufuncs (linear) -----
def_linear(np.multiply)

# ----- Binary ufuncs -----
defjvp(
    np.add,
    lambda ans, x, y: lambda g: np.broadcast_to(g, np.shape(ans)),
    lambda ans, x, y: lambda g: np.broadcast_to(g, np.shape(ans)),
)
defjvp(
    np.subtract,
    lambda ans, x, y: lambda g: np.broadcast_to(g, np.shape(ans)),
    lambda ans, x, y: lambda g: np.broadcast_to(-g, np.shape(ans)),
)
defjvp(
    np.multiply,
    lambda ans, x, y: lambda g: np.broadcast_to(g * y, np.shape(ans)),
    lambda ans, x, y: lambda g: np.broadcast_to(x * g, np.shape(ans)),
)
defjvp(np.divide, "same", lambda ans, x, y: lambda g: -g * x / y ** 2)
defjvp(
    np.maximum,
    lambda ans, x, y: lambda g: g * balanced_eq(x, ans, y),
    lambda ans, x, y: lambda g: g * balanced_eq(y, ans, x),
)
defjvp(
    np.minimum,
    lambda ans, x, y: lambda g: g * balanced_eq(x, ans, y),
    lambda ans, x, y: lambda g: g * balanced_eq(y, ans, x),
)
defjvp(
    np.fmax,
    lambda ans, x, y: lambda g: g * balanced_eq(x, ans, y),
    lambda ans, x, y: lambda g: g * balanced_eq(y, ans, x),
)
defjvp(
    np.fmin,
    lambda ans, x, y: lambda g: g * balanced_eq(x, ans, y),
    lambda ans, x, y: lambda g: g * balanced_eq(y, ans, x),
)
defjvp(
    np.logaddexp,
    lambda ans, x, y: lambda g: g * np.exp(x - ans),
    lambda ans, x, y: lambda g: g * np.exp(y - ans),
)
defjvp(
    np.logaddexp2,
    lambda ans, x, y: lambda g: g * 2 ** (x - ans),
    lambda ans, x, y: lambda g: g * 2 ** (y - ans),
)
defjvp(np.true_divide, "same", lambda ans, x, y: lambda g: -g * x / y ** 2)
defjvp(
    np.mod,
    lambda ans, x, y: lambda g: np.broadcast_to(g, np.shape(ans)),
    lambda ans, x, y: lambda g: -g * np.floor(x / y),
)
defjvp(
    np.remainder,
    lambda ans, x, y: lambda g: np.broadcast_to(g, np.shape(ans)),
    lambda ans, x, y: lambda g: -g * np.floor(x / y),
)
defjvp(
    np.power,
    lambda ans, x, y: lambda g: g * y * x ** np.where(y, y - 1, 1.0),
    lambda ans, x, y: lambda g: g * np.log(replace_zero(x, 1.0)) * ans,
)
defjvp(
    np.arctan2,
    lambda ans, x, y: lambda g: g * y / (x ** 2 + y ** 2),
    lambda ans, x, y: lambda g: g * -x / (x ** 2 + y ** 2),
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
defjvp(np.flipud, "same")
defjvp(np.fliplr, "same")
defjvp(np.rot90, "same")
defjvp(np.full, "same", argnums=(1,))
defjvp(np.triu, "same")
defjvp(np.tril, "same")
defjvp(np.swapaxes, "same")
defjvp(np.rollaxis, "same")
defjvp(np.moveaxis, "same")
defjvp(np.broadcast_to, "same")
def_linear(np.cross)

# ----- Simple grads -----
defjvp(np.positive, lambda ans, x: lambda g: np.ones_like(x) * g)
defjvp(np.negative, lambda ans, x: lambda g: -np.ones_like(x) * g)
defjvp(
    np.fabs, lambda ans, x: lambda g: np.sign(x) * g
)  # fabs doesn't take complex numbers.
defjvp(np.absolute, lambda ans, x: lambda g: np.real(g * np.conj(x)) / ans)
defjvp(np.reciprocal, lambda ans, x: lambda g: -g / x ** 2)
defjvp(np.exp, lambda ans, x: lambda g: ans * g)
defjvp(np.exp2, lambda ans, x: lambda g: ans * np.log(2) * g)
defjvp(np.expm1, lambda ans, x: lambda g: (ans + 1) * g)
defjvp(np.log, lambda ans, x: lambda g: g / x)
defjvp(np.log2, lambda ans, x: lambda g: g / x / np.log(2))
defjvp(np.log10, lambda ans, x: lambda g: g / x / np.log(10))
defjvp(np.log1p, lambda ans, x: lambda g: g / (x + 1))
defjvp(np.sin, lambda ans, x: lambda g: g * np.cos(x))
defjvp(np.cos, lambda ans, x: lambda g: -g * np.sin(x))
defjvp(np.tan, lambda ans, x: lambda g: g / np.cos(x) ** 2)
defjvp(np.arcsin, lambda ans, x: lambda g: g / np.sqrt(1 - x ** 2))
defjvp(np.arccos, lambda ans, x: lambda g: -g / np.sqrt(1 - x ** 2))
defjvp(np.arctan, lambda ans, x: lambda g: g / (1 + x ** 2))
defjvp(np.sinh, lambda ans, x: lambda g: g * np.cosh(x))
defjvp(np.cosh, lambda ans, x: lambda g: g * np.sinh(x))
defjvp(np.tanh, lambda ans, x: lambda g: g / np.cosh(x) ** 2)
defjvp(np.arcsinh, lambda ans, x: lambda g: g / np.sqrt(x ** 2 + 1))
defjvp(np.arccosh, lambda ans, x: lambda g: g / np.sqrt(x ** 2 - 1))
defjvp(np.arctanh, lambda ans, x: lambda g: g / (1 - x ** 2))
defjvp(np.square, lambda ans, x: lambda g: g * 2 * x)
defjvp(np.sqrt, lambda ans, x: lambda g: g * 0.5 * x ** -0.5)
defjvp(
    np.sinc,
    lambda ans, x: lambda g: g
    * (np.cos(np.pi * x) * np.pi * x - np.sin(np.pi * x))
    / (np.pi * x ** 2),
)
defjvp(
    np.clip,
    lambda ans, x, a_min, a_max: lambda g: g
    * np.logical_and(ans != a_min, ans != a_max),
)
defjvp(np.real_if_close, lambda ans, x: lambda g: match_complex(ans, g))
defjvp(np.real, lambda ans, x: lambda g: np.real(g))
defjvp(np.imag, lambda ans, x: lambda g: match_complex(ans, -1j * g))
defjvp(np.conj, lambda ans, x: lambda g: np.conj(g))
defjvp(
    np.angle,
    lambda ans, x: lambda g: match_complex(ans, g * np.conj(x * 1j) / np.abs(x) ** 2),
)
defjvp(
    np.where,
    None,
    lambda ans, c, x=None, y=None: lambda g: np.where(c, g, np.zeros(np.shape(g))),
    lambda ans, c, x=None, y=None: lambda g: np.where(c, np.zeros(g.shape), g),
)

# ----- Trickier grads -----
# defjvp(np.kron, "same", "same")
defjvp(np.diff, "same")
defjvp(np.gradient, "same")
defjvp(np.repeat, "same")
defjvp(np.tile, "same")
defjvp(np.transpose, "same")
defjvp(np.sum, "same")

defjvp(
    np.prod,
    lambda ans, x, axis=None, keepdims=False: lambda g: ans
    * np.sum(g / x, axis=axis, keepdims=keepdims),
)
defjvp(
    np.linspace,
    lambda ans, start, stop, *args, **kwargs: lambda g: np.linspace(
        g, 0, *args, **kwargs
    ),
    lambda ans, start, stop, *args, **kwargs: lambda g: np.linspace(
        0, g, *args, **kwargs
    ),
)


def forward_grad_np_var(ans, x, axis=None, ddof=0, keepdims=False):
    def jvp(g):
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

    return jvp


defjvp(np.var, forward_grad_np_var)


def forward_grad_np_std(ans, x, axis=None, ddof=0, keepdims=False):
    def jvp(g):
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

    return jvp


defjvp(np.std, forward_grad_np_std)


def fwd_grad_chooser(ans, x, axis=None, keepdims=False):
    def jvp(g):
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

    return jvp


defjvp(np.max, fwd_grad_chooser)
defjvp(np.min, fwd_grad_chooser)


defjvp(np.cumsum, "same")

def_linear(np.matmul)


def fwd_grad_sort(g, ans, x, axis=-1, kind="quicksort", order=None):
    sort_perm = np.argsort(x, axis, kind, order)
    return g[sort_perm]


defjvp(np.sort, fwd_grad_sort)
defjvp(np.msort, lambda ans, x: lambda g: fwd_grad_sort(g, ans, x, axis=0))


def fwd_grad_partition(ans, x, kth, axis=-1, kind="introselect", order=None):
    def jvp(g):
        partition_perm = np.argpartition(x, kth, axis, kind, order)
        return g[partition_perm]

    return jvp


defjvp(np.partition, fwd_grad_partition)


def atleast_jvpmaker(fun):
    def jvp(g, ans, *arys):
        if len(arys) > 1:
            raise NotImplementedError("Can't handle multiple arguments yet.")
        return lambda g: fun(g)

    return jvp


defjvp(np.atleast_1d, atleast_jvpmaker(np.atleast_1d))
defjvp(np.atleast_2d, atleast_jvpmaker(np.atleast_2d))
defjvp(np.atleast_3d, atleast_jvpmaker(np.atleast_3d))


defjvp(
    np.pad, lambda ans, array, width, mode, **kwargs: lambda g: np.pad(g, width, mode)
)


def stack_diff(ans, x, axis=0):
    def jvp(g):
        ret = []
        ng = np.broadcast_to(g, np.shape(ans))
        shape = np.shape(ng)
        for idx in range(shape[axis]):
            ret.append(np.take(ng, idx, axis=axis))
        return tuple(ret)

    return jvp


defjvp(np.stack, stack_diff)
