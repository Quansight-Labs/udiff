from ._vjp_core import defvjp
import unumpy as np
from functools import partial

# ----- Non-differentiable functions -----

nograd_functions = set(
    [
        np.array,
        np.ones,
        np.zeros,
        np.floor,
        np.ceil,
        np.rint,
        np.trunc,
        np.all,
        np.arange,
        np.any,
        np.argmax,
        np.argmin,
        np.argpartition,
        np.argsort,
        np.argwhere,
        np.nonzero,
        np.asarray,
        np.flatnonzero,
        np.count_nonzero,
        np.searchsorted,
        np.sign,
        np.ndim,
        np.shape,
        np.dtype,
        np.floor_divide,
        np.logical_and,
        np.logical_or,
        np.logical_not,
        np.logical_xor,
        np.isfinite,
        np.isinf,
        np.isnan,
        np.equal,
        np.not_equal,
        np.size,
        # np.array_equiv,
        np.greater,
        np.greater_equal,
        np.less,
        np.less_equal,
        # np.round,
        # np.around,
        # np.fix,
        # np.isneginf,
        # np.isposinf,
        # np.allclose,
        # np.isclose,
        # np.array_equal,
        # np.iscomplexobj,
        # np.iscomplex,
        # np.isscalar,
        # np.isreal,
        # np.zeros_like,
        # np.ones_like,
        # np.result_type,
    ]
)

# defvjp(np.nan_to_num, lambda ans, x: lambda g: np.where(np.isfinite(x), g, 0.))

# ----- Binary ufuncs -----

defvjp(
    np.add,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g),
)
defvjp(
    np.multiply,
    lambda ans, x, y: unbroadcast_f(x, lambda g: y * g),
    lambda ans, x, y: unbroadcast_f(y, lambda g: x * g),
)
defvjp(
    np.subtract,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g),
    lambda ans, x, y: unbroadcast_f(y, lambda g: -g),
)
defvjp(
    np.divide,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g / y),
    lambda ans, x, y: unbroadcast_f(y, lambda g: -g * x / y ** 2),
)
defvjp(
    np.maximum,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
defvjp(
    np.minimum,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
defvjp(
    np.fmax,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
defvjp(
    np.fmin,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
defvjp(
    np.logaddexp,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * np.exp(x - ans)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * np.exp(y - ans)),
)
defvjp(
    np.logaddexp2,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * 2 ** (x - ans)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * 2 ** (y - ans)),
)
defvjp(
    np.true_divide,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g / y),
    lambda ans, x, y: unbroadcast_f(y, lambda g: -g * x / y ** 2),
)
defvjp(
    np.mod,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g),
    lambda ans, x, y: unbroadcast_f(y, lambda g: -g * np.floor(x / y)),
)
defvjp(
    np.remainder,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g),
    lambda ans, x, y: unbroadcast_f(y, lambda g: -g * np.floor(x / y)),
)
defvjp(
    np.power,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * y * x ** np.where(y, y - 1, 1.0)),
    lambda ans, x, y: unbroadcast_f(
        y, lambda g: g * np.log(replace_non_positive(x, 1.0)) * ans
    ),
)
defvjp(
    np.arctan2,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * y / (x ** 2 + y ** 2)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * -x / (x ** 2 + y ** 2)),
)
defvjp(
    np.hypot,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * x / ans),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * y / ans),
)

# ----- Simple grads -----

defvjp(np.positive, lambda ans, x: lambda g: g)
defvjp(np.negative, lambda ans, x: lambda g: -g)
defvjp(
    np.absolute,
    lambda ans, x: lambda g: g * replace_zero(np.conj(x), 0.0) / replace_zero(ans, 1.0),
)
defvjp(
    np.fabs, lambda ans, x: lambda g: np.sign(x) * g
)  # fabs doesn't take complex numbers.
defvjp(np.absolute, lambda ans, x: lambda g: g * np.conj(x) / ans)
defvjp(np.reciprocal, lambda ans, x: lambda g: -g / x ** 2)
defvjp(np.exp, lambda ans, x: lambda g: ans * g)
defvjp(np.exp2, lambda ans, x: lambda g: ans * np.log(2) * g)
defvjp(np.expm1, lambda ans, x: lambda g: (ans + 1) * g)
defvjp(np.log, lambda ans, x: lambda g: g / x)
defvjp(np.log2, lambda ans, x: lambda g: g / x / np.log(2))
defvjp(np.log10, lambda ans, x: lambda g: g / x / np.log(10))
defvjp(np.log1p, lambda ans, x: lambda g: g / (x + 1))
defvjp(np.sin, lambda ans, x: lambda g: g * np.cos(x))
defvjp(np.cos, lambda ans, x: lambda g: -g * np.sin(x))
defvjp(np.tan, lambda ans, x: lambda g: g / np.cos(x) ** 2)
defvjp(np.arcsin, lambda ans, x: lambda g: g / np.sqrt(1 - x ** 2))
defvjp(np.arccos, lambda ans, x: lambda g: -g / np.sqrt(1 - x ** 2))
defvjp(np.arctan, lambda ans, x: lambda g: g / (1 + x ** 2))
defvjp(np.sinh, lambda ans, x: lambda g: g * np.cosh(x))
defvjp(np.cosh, lambda ans, x: lambda g: g * np.sinh(x))
defvjp(np.tanh, lambda ans, x: lambda g: g / np.cosh(x) ** 2)
defvjp(np.arcsinh, lambda ans, x: lambda g: g / np.sqrt(x ** 2 + 1))
defvjp(np.arccosh, lambda ans, x: lambda g: g / np.sqrt(x ** 2 - 1))
defvjp(np.arctanh, lambda ans, x: lambda g: g / (1 - x ** 2))
defvjp(np.rad2deg, lambda ans, x: lambda g: g / np.pi * 180.0)
# defvjp(np.degrees, lambda ans, x : lambda g: g / np.pi * 180.0)
defvjp(np.deg2rad, lambda ans, x: lambda g: g * np.pi / 180.0)
# defvjp(np.radians, lambda ans, x : lambda g: g * np.pi / 180.0)
defvjp(np.square, lambda ans, x: lambda g: g * 2 * x)
defvjp(np.sqrt, lambda ans, x: lambda g: g * 0.5 * x ** -0.5)
# defvjp(np.sinc,    lambda ans, x : lambda g: g * (np.cos(np.pi*x)*np.pi*x - np.sin(np.pi*x))/(np.pi*x**2))
defvjp(
    np.reshape,
    lambda ans, x, shape, order=None: lambda g: np.reshape(g, np.shape(x), order=order),
)
# defvjp(np.roll,    lambda ans, x, shift, axis=None  : lambda g: np.roll(g, -shift, axis=axis))
# defvjp(np.array_split, lambda ans, ary, idxs, axis=0 : lambda g: np.concatenate(g, axis=axis))
# defvjp(np.split,       lambda ans, ary, idxs, axis=0 : lambda g: np.concatenate(g, axis=axis))
# defvjp(np.vsplit,      lambda ans, ary, idxs         : lambda g: np.concatenate(g, axis=0))
# defvjp(np.hsplit,      lambda ans, ary, idxs         : lambda g: np.concatenate(g, axis=1))
# defvjp(np.dsplit,      lambda ans, ary, idxs         : lambda g: np.concatenate(g, axis=2))
defvjp(
    np.ravel,
    lambda ans, x, order=None: lambda g: np.reshape(g, np.shape(x), order=order),
)
# defvjp(np.expand_dims, lambda ans, x, axis     : lambda g: np.reshape(g, np.shape(x)))
# defvjp(np.squeeze, lambda ans, x, axis=None    : lambda g: np.reshape(g, np.shape(x)))
# defvjp(np.diag,    lambda ans, x, k=0          : lambda g: np.diag(g, k))
# defvjp(np.flipud,  lambda ans, x,              : lambda g: np.flipud(g))
# defvjp(np.fliplr,  lambda ans, x,              : lambda g: np.fliplr(g))
# defvjp(np.rot90,   lambda ans, x, k=1          : lambda g: np.rot90(g, -k))
# defvjp(np.trace,   lambda ans, x, offset=0     : lambda g: np.einsum('ij,...->ij...', np.eye(x.shape[0], x.shape[1], k=offset), g))
defvjp(
    np.full,
    lambda ans, shape, fill_value, dtype=None: lambda g: np.sum(g),
    argnums=(1,),
)
# defvjp(np.triu,    lambda ans, x, k=0          : lambda g: np.triu(g, k=k))
# defvjp(np.tril,    lambda ans, x, k=0          : lambda g: np.tril(g, k=k))
# defvjp(np.clip,    lambda ans, x, a_min, a_max : lambda g: g * np.logical_and(ans != a_min, ans != a_max))
defvjp(np.swapaxes, lambda ans, x, axis1, axis2: lambda g: np.swapaxes(g, axis2, axis1))
defvjp(
    np.moveaxis,
    lambda ans, a, source, destination: lambda g: np.moveaxis(g, destination, source),
)
# defvjp(np.real_if_close, lambda ans, x : lambda g: match_complex(x, g))
# defvjp(np.real,   lambda ans, x   : lambda g: match_complex(x, g))
# defvjp(np.imag,   lambda ans, x   : lambda g: match_complex(x, -1j * g))
defvjp(np.conj, lambda ans, x: lambda g: np.conj(g))
# defvjp(np.conjugate, lambda ans, x: lambda g: np.conj(g))
# defvjp(np.angle,  lambda ans, x   : lambda g: match_complex(x, g * np.conj(x * 1j) / np.abs(x)**2))
defvjp(
    np.where,
    None,
    # lambda ans, c, x=None, y=None: lambda g: np.where(c, g, np.zeros_like(g)),
    lambda ans, c, x=None, y=None: lambda g: np.where(c, g, np.zeros(np.shape(g))),
    # lambda ans, c, x=None, y=None: lambda g: np.where(c, np.zeros_like(g), g),
    lambda ans, c, x=None, y=None: lambda g: np.where(c, np.zeros(np.shape(g)), g),
)
# defvjp(np.cross, lambda ans, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None : lambda g:
#                   np.cross(b, g, axisb, axisc, axisa, axis),
#                   lambda ans, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None : lambda g:
#                   np.cross(g, a, axisc, axisa, axisb, axis))
# defvjp(np.linspace, lambda ans, start, stop, num : lambda g: np.dot(np.linspace(1.0, 0.0, num), g),
#                      lambda ans, start, stop, num : lambda g: np.dot(np.linspace(0.0, 1.0, num), g))

# defvjp(np._astype,
#        lambda ans, A, dtype, order='K', casting='unsafe', subok=True, copy=True:
#        lambda g: np._astype(g, A.dtype))

# ----- Trickier grads -----
def grad_rollaxis(ans, a, axis, start=0):
    if axis < 0:
        raise NotImplementedError(
            "Gradient of rollaxis not implemented for axis < 0. "
            "Please use moveaxis instead."
        )
    elif start < 0:
        raise NotImplementedError(
            "Gradient of rollaxis not implemented for start < 0. "
            "Please use moveaxis instead."
        )
    return (
        lambda g: np.rollaxis(g, start - 1, axis)
        if start > axis
        else np.rollaxis(g, start, axis + 1)
    )


defvjp(np.rollaxis, grad_rollaxis)

# def grad_diff(ans, a, n=1, axis=-1):
#     nd = np.ndim(a)
#     ans_shape = np.shape(ans)
#     sl1 = [slice(None)]*nd
#     sl1[axis] = slice(None, 1)

#     sl2 = [slice(None)]*nd
#     sl2[axis] = slice(-1, None)

#     def undiff(g):
#         if g.shape[axis] > 0:
#             return np.concatenate((-g[tuple(sl1)], -np.diff(g, axis=axis), g[tuple(sl2)]), axis=axis)
#         shape = list(ans_shape)
#         shape[axis] = 1
#         return np.zeros(shape)

#     def helper(g, n):
#         if n == 0:
#             return g
#         return helper(undiff(g), n-1)
#     return lambda g: helper(g, n)

# defvjp(np.diff, grad_diff)

# def grad_gradient(ans, x, *vargs, **kwargs):
#     axis = kwargs.pop('axis', None)
#     if vargs or kwargs:
#         raise NotImplementedError(
#             "The only optional argument currently supported for np.gradient "
#             "is axis.")
#     if axis is None:
#         axis = range(x.ndim)
#     elif type(axis) is int:
#         axis = [axis]
#     else:
#         axis = list(axis)

#     x_dtype = x.dtype
#     x_shape = x.shape
#     nd = x.ndim

#     def vjp(g):
#         if np.ndim(g) == nd:
#             # add axis if gradient was along one axis only
#             g = g[np.newaxis]

#         # accumulate gradient
#         out = np.zeros(x_shape, dtype=x_dtype)

#         for i, a in enumerate(axis):
#             # swap gradient axis to the front
#             g_swap = np.swapaxes(g[i], 0, a)[:, np.newaxis]

#             out_axis = np.concatenate((
#                 -g_swap[0] - 0.5 * g_swap[1],
#                  g_swap[0] - 0.5 * g_swap[2],
#                 (-1.) * np.gradient(g_swap, axis=0)[2:-2, 0],
#                 0.5 * g_swap[-3] - g_swap[-1],
#                 0.5 * g_swap[-2] + g_swap[-1],
#             ), axis=0)

#             out = out + np.swapaxes(out_axis, 0, a)

#         return out

#     return vjp

# defvjp(np.gradient, grad_gradient)

# def grad_repeat(ans, x, repeats, axis=None):
#     shape = np.shape(x)
#     def vjp(g):
#         if axis is None:  # If axis is none, np.repeat() repeats the flattened array.
#             expanded = np.reshape(g, (np.prod(shape),) + (repeats,))
#             return np.reshape(np.sum(expanded, axis=1, keepdims=False), shape)
#         else:
#             if shape[axis] == 1:  # For this common case, the logic is simple.
#                 return np.sum(g, axis=axis, keepdims=True)
#             else:
#                 expanded = np.reshape(g, shape[0:axis+1] + (repeats,) + shape[axis+1:])
#                 return np.sum(expanded, axis=axis+1, keepdims=False)
#     return vjp

# defvjp(np.repeat, grad_repeat)

# def grad_tile(ans, x, reps):
#     reps = [reps] if np.isscalar(reps) else reps
#     x_shape = np.shape(x)
#     def vjp(g):
#         for axis, rep in enumerate(reps):
#             g = sum(np.split(g, rep, axis))
#         return np.reshape(g, x_shape)
#     return vjp
# defvjp(np.tile, grad_tile)

# def grad_kron(argnum, ans, orig_A, orig_B):
#     # kron has different promotion rules than dot. the reshapes are necessary if
#     # and only if (1) orig_B is 1D or (2) orig_A and/or orig_B are 0D
#     orig_A_shape = np.shape(orig_A)
#     orig_B_shape = np.shape(orig_B)
#     def vjp(G):
#         A, B = np.atleast_2d(orig_A), np.atleast_2d(orig_B)
#         shape = list(A.shape + B.shape)
#         n = np.ndim(A)
#         shape[n-1], shape[n] = shape[n], shape[n-1]
#         reshaped_G = np.swapaxes(np.reshape(G, shape), n-1, n)
#         if argnum == 0:
#             return match_complex(orig_A, np.reshape(np.tensordot(reshaped_G, B, axes=np.ndim(B)), orig_A_shape))
#         else:
#             return match_complex(orig_B, np.reshape(np.tensordot(A, reshaped_G, axes=np.ndim(A)), orig_B_shape))
#     return vjp
# defvjp(np.kron, partial(grad_kron, 0), partial(grad_kron, 1))


def grad_transpose(ans, x, axes=None):
    if axes is not None:
        axes = np.argsort(axes)
    return lambda g: np.transpose(g, axes)


defvjp(np.transpose, grad_transpose)


def repeat_to_match_shape(g, shape, dtype, axis, keepdims):
    """Returns the array g repeated along axis to fit vector space vs.
       Also returns the number of repetitions of the array."""
    if shape == ():
        return g, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = np.array(shape, dtype=int)
    new_shape[axis] = 1
    num_reps = np.prod(np.array(shape)[axis])
    # Can't use broadcast_to because of numpy bug: https://github.com/numpy/numpy/issues/9165
    # return anp.broadcast_to(anp.reshape(g, new_shape), shape), num_reps
    return np.reshape(g, new_shape) + np.zeros(shape, dtype=dtype), num_reps


def grad_broadcast_to(ans, x, new_shape):
    old_shape = np.shape(x)
    assert np.shape(ans) == new_shape
    assert len(old_shape) == len(new_shape), "Can't handle extra leading dims"
    broadcast_axes = tuple(
        np.where(np.logical_and(np.array(old_shape) == 1, np.array(new_shape) > 1))[0]
    )
    return lambda g: np.sum(g, axis=broadcast_axes, keepdims=True)


defvjp(np.broadcast_to, grad_broadcast_to)


def grad_np_sum(ans, x, axis=None, keepdims=False, dtype=None):
    shape, dtype = np.shape(x), x.dtype
    return lambda g: repeat_to_match_shape(g, shape, dtype, axis, keepdims)[0]


defvjp(np.sum, grad_np_sum)


# def grad_np_mean(ans, x, axis=None, keepdims=False):
#     shape, dtype = np.shape(x), np.result_type(x)
#     def vjp(g):
#         g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
#         return g_repeated / num_reps
#     return vjp
# defvjp(np.mean, grad_np_mean)


def grad_np_prod(ans, x, axis=None, keepdims=False):  # TODO: Support tuples of axes.
    shape, dtype = np.shape(x), np.result_type(x)

    def vjp(g):
        g_repeated, _ = repeat_to_match_shape(g * ans, shape, dtype, axis, keepdims)
        return g_repeated / x

    return vjp


defvjp(np.prod, grad_np_prod)


def grad_np_var(ans, x, axis=None, ddof=0, keepdims=False):
    shape, _, dtype, iscomplex = metadata(x)

    def vjp(g):
        if iscomplex:
            g = g + 0j
        g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
        x_minus_mean = np.conj(x - np.mean(x, axis=axis, keepdims=True))
        return 2.0 * g_repeated * x_minus_mean / (num_reps - ddof)

    return vjp


defvjp(np.var, grad_np_var)


def grad_np_std(ans, x, axis=None, ddof=0, keepdims=False):
    shape, _, dtype, iscomplex = metadata(x)

    def vjp(g):
        if iscomplex:
            g = g + 0j
        g_repeated, num_reps = repeat_to_match_shape(
            g, shape, dtype, axis, keepdims
        )  # Avoid division by zero.
        if num_reps <= 1:
            return g_repeated * 0.0
        else:
            g_repeated, num_reps = repeat_to_match_shape(
                g / ans, shape, dtype, axis, keepdims
            )
            x_minus_mean = np.conj(x - np.mean(x, axis=axis, keepdims=True))
            return g_repeated * x_minus_mean / (num_reps - ddof)

    return vjp


defvjp(np.std, grad_np_std)


def grad_chooser(ans, x, axis=None, keepdims=None):
    shape, dtype = np.shape(x), np.result_type(x)

    def vjp(g):
        """Builds gradient of functions that choose a single item, such as min or max."""
        g_repeated, _ = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
        argmax_locations = (
            x == repeat_to_match_shape(ans, shape, dtype, axis, keepdims)[0]
        )
        return (
            g_repeated
            * argmax_locations
            / np.sum(argmax_locations, axis=axis, keepdims=True)
        )

    return vjp


defvjp(np.max, grad_chooser)
defvjp(np.min, grad_chooser)
# defvjp(np.amax, grad_chooser)
# defvjp(np.amin, grad_chooser)


def reverse_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1, ...]
    return x.swapaxes(0, axis)


# def grad_np_cumsum(ans, x, axis=None):
#     def vjp(g):
#         if axis:
#             return reverse_axis(np.cumsum(reverse_axis(g, axis), axis), axis)
#         else:
#             return np.reshape(np.cumsum(g[::-1], axis)[::-1], x.shape)
#     return vjp
# defvjp(np.cumsum, grad_np_cumsum)

# def grad_inner(argnum, ans, A, B):
#     A_ndim, B_ndim = np.ndim(A), np.ndim(B)
#     if A_ndim == 0 or B_ndim == 0:
#         axes = ([], [])
#     else:
#         axes = ([A_ndim - 1], [B_ndim - 1])
#     if argnum == 0:
#         return lambda G: tensordot_adjoint_0(B, G, axes, A_ndim, B_ndim)
#     elif argnum == 1:
#         return lambda G: tensordot_adjoint_1(A, G, axes, A_ndim, B_ndim)
# defvjp(np.inner, partial(grad_inner, 0), partial(grad_inner, 1))


def matmul_adjoint_0(B, G, A_meta, B_ndim):
    if np.ndim(G) == 0:  # A_ndim == B_ndim == 1
        return unbroadcast(G * B, A_meta)
    _, A_ndim, _, = A_meta
    if A_ndim == 1:
        G = np.expand_dims(G, np.ndim(G) - 1)
    if B_ndim == 1:  # The result we need is an outer product
        B = np.expand_dims(B, 0)
        G = np.expand_dims(G, np.ndim(G))
    else:  # We need to swap the last two axes of B
        B = np.swapaxes(B, B_ndim - 2, B_ndim - 1)
    result = np.matmul(G, B)
    return unbroadcast(result, A_meta)


def matmul_adjoint_1(A, G, A_ndim, B_meta):
    if np.ndim(G) == 0:  # A_ndim == B_ndim == 1
        return unbroadcast(G * A, B_meta)
    _, B_ndim, _ = B_meta
    B_is_vec = B_ndim == 1
    if B_is_vec:
        G = np.expand_dims(G, np.ndim(G))
    if A_ndim == 1:  # The result we need is an outer product
        A = np.expand_dims(A, 1)
        G = np.expand_dims(G, np.ndim(G) - 1)
    else:  # We need to swap the last two axes of A
        A = np.swapaxes(A, A_ndim - 2, A_ndim - 1)
    result = np.matmul(A, G)
    if B_is_vec:
        result = np.squeeze(result, np.ndim(G) - 1)
    return unbroadcast(result, B_meta)


def matmul_vjp_0(ans, A, B):
    A_meta = metadata(A)
    B_ndim = np.ndim(B)
    return lambda g: matmul_adjoint_0(B, g, A_meta, B_ndim)


def matmul_vjp_1(ans, A, B):
    A_ndim = np.ndim(A)
    B_meta = metadata(B)
    return lambda g: matmul_adjoint_1(A, g, A_ndim, B_meta)


defvjp(np.matmul, matmul_vjp_0, matmul_vjp_1)

# @primitive
# def dot_adjoint_0(B, G, A_meta, B_meta):
#     _, A_ndim, A_dtype, _ = A_meta
#     _, B_ndim, _, _ = B_meta
#     if B_ndim == 0 or B_ndim == 1 or A_ndim == 0:
#         contract_num = max(0, B_ndim - (A_ndim != 0))
#         out = onp.tensordot(G, B, contract_num)
#     else:
#         out = onp.tensordot(G, onp.swapaxes(B, -1, -2), B_ndim - 1)
#     return onp.asarray(out, dtype=A_dtype)

# @primitive
# def dot_adjoint_1(A, G, A_meta, B_meta):
#     _, A_ndim, _, _ = A_meta
#     _, B_ndim, B_dtype, _ = B_meta
#     needs_transpose = B_ndim > 1 and A_ndim != 0
#     swap = (lambda x: onp.swapaxes(x, -1, -2)) if needs_transpose else (lambda x: x)
#     if A_ndim == 0 or A_ndim == 1 or B_ndim == 0:
#         contract_num = max(0, A_ndim - (B_ndim != 0))
#         out = swap(onp.tensordot(G, A, contract_num))
#     else:
#         out = swap(onp.tensordot(
#             G, A, [range(-A_ndim - B_ndim + 2, -B_ndim + 1), range(A_ndim - 1)]))
#     return onp.asarray(out, dtype=B_dtype)

# def dot_vjp_0(ans, A, B):
#     A_meta, B_meta = metadata(A), metadata(B)
#     return lambda g: match_complex(A, dot_adjoint_0(B, g, A_meta, B_meta))

# def dot_vjp_1(ans, A, B):
#     A_meta, B_meta = metadata(A), metadata(B)
#     return lambda g: match_complex(B, dot_adjoint_1(A, g, A_meta, B_meta))
# defvjp(np.dot, dot_vjp_0, dot_vjp_1)

# defvjp(dot_adjoint_0, lambda ans, B, g, An, Bn: lambda A: match_complex(B, dot_adjoint_1(A, g, An, Bn)),
#                       lambda ans, B, g, An, Bn: lambda A: match_complex(g, np.dot(A, B)))

# defvjp(dot_adjoint_1, lambda ans, A, g, An, Bn: lambda B: match_complex(A, dot_adjoint_0(B, g, An, Bn)),
#                       lambda ans, A, g, An, Bn: lambda B: match_complex(g, np.dot(A, B)))

# @primitive
# def tensordot_adjoint_0(B, G, axes, A_ndim, B_ndim):
#     # The adjoint of the operator
#     # A |--> np.tensordot(A, B, axes)
#     if B_ndim == 0:
#         return G * B

#     G_axes = onp.arange(onp.ndim(G))
#     if type(axes) is int:
#         axes = max(axes, 0)
#         B_axes = onp.arange(B_ndim)
#         return onp.tensordot(G, B, [G_axes[A_ndim-axes:], B_axes[axes:]])
#     else:
#         axes0 = [axes[0]] if type(axes[0]) is int else axes[0]
#         axes1 = [axes[1]] if type(axes[1]) is int else axes[1]
#         axes = [axes0, axes1]
#         A_axes = onp.arange(A_ndim)
#         B_axes = onp.arange(B_ndim)
#         summed_axes = [onp.asarray(axes[0], dtype='int64') % A_ndim,
#                        onp.asarray(axes[1], dtype='int64') % B_ndim]
#         other_axes  = [onp.delete(A_axes, summed_axes[0]),
#                        onp.delete(B_axes, summed_axes[1])]
#         out = onp.tensordot(G, B, [G_axes[len(other_axes[0]):], other_axes[1]])
#         perm = onp.argsort(onp.concatenate(
#             (other_axes[0], summed_axes[0][onp.argsort(summed_axes[1])])))
#         return onp.transpose(out, perm)

# @primitive
# def tensordot_adjoint_1(A, G, axes, A_ndim, B_ndim):
#     # The adjoint of the operator
#     # B |--> np.tensordot(A, B, axes)
#     if A_ndim == 0:
#         return G * A

#     G_axes = onp.arange(onp.ndim(G))
#     if type(axes) is int:
#         axes = max(axes, 0)
#         A_axes = onp.arange(A_ndim)
#         return onp.tensordot(A, G, [A_axes[:A_ndim-axes], G_axes[:A_ndim-axes]])
#     else:
#         axes0 = [axes[0]] if type(axes[0]) is int else axes[0]
#         axes1 = [axes[1]] if type(axes[1]) is int else axes[1]
#         axes = [axes0, axes1]
#         A_axes = onp.arange(A_ndim)
#         B_axes = onp.arange(B_ndim)
#         summed_axes = [onp.asarray(axes[0], dtype='int64') % A_ndim,
#                        onp.asarray(axes[1], dtype='int64') % B_ndim]
#         other_axes  = [onp.delete(A_axes, summed_axes[0]),
#                        onp.delete(B_axes, summed_axes[1])]
#         out = onp.tensordot(A, G, [other_axes[0], G_axes[:len(other_axes[0])]])
#         perm = onp.argsort(onp.concatenate(
#             (summed_axes[1][onp.argsort(summed_axes[0])], other_axes[1])))
#         return onp.transpose(out, perm)

# def tensordot_vjp_0(ans, A, B, axes=2):
#     A_ndim, B_ndim = np.ndim(A), np.ndim(B)
#     return lambda G: match_complex(A, tensordot_adjoint_0(B, G, axes, A_ndim, B_ndim))

# def tensordot_vjp_1(ans, A, B, axes=2):
#     A_ndim, B_ndim = np.ndim(A), np.ndim(B)
#     return lambda G: match_complex(B, tensordot_adjoint_1(A, G, axes, A_ndim, B_ndim))

# defvjp(np.tensordot, tensordot_vjp_0, tensordot_vjp_1)
# defvjp(tensordot_adjoint_0, lambda ans, B, G, axes, An, Bn: lambda A: match_complex(B, tensordot_adjoint_1(A, G, axes, An, Bn)),
#                             lambda ans, B, G, axes, An, Bn: lambda A: match_complex(G, np.tensordot(A, B, axes)))
# defvjp(tensordot_adjoint_1, lambda ans, A, G, axes, An, Bn: lambda B: match_complex(A, tensordot_adjoint_0(B, G, axes, An, Bn)),
#                             lambda ans, A, G, axes, An, Bn: lambda B: match_complex(G, np.tensordot(A, B, axes)))
# defvjp(np.outer, lambda ans, a, b : lambda g: match_complex(a, np.dot(g, b.T)),
#                   lambda ans, a, b : lambda g: match_complex(b, np.dot(a.T, g)))

# def grad_concatenate_args(argnum, ans, axis_args, kwargs):
#     axis, args = axis_args[0], axis_args[1:]
#     sizes = [np.shape(a)[axis] for a in args[:argnum]]
#     start = sum(sizes[:-1])
#     idxs = [slice(None)] * ans.ndim
#     idxs[axis] = slice(start, start + sizes[-1])
#     return lambda g: g[tuple(idxs)]
# defvjp_argnum(np.concatenate_args, grad_concatenate_args)


def grad_sort(ans, x, axis=-1, kind="quicksort", order=None):
    # TODO: Cast input with np.asanyarray()
    if len(np.shape(x)) > 1:
        raise NotImplementedError(
            "Gradient of sort not implemented for multi-dimensional arrays."
        )
    sort_perm = np.argsort(x, axis, kind, order)
    return lambda g: unpermuter(g, sort_perm)


defvjp(np.sort, grad_sort)
defvjp(np.msort, grad_sort)  # Until multi-D is allowed, these are the same.


def grad_partition(ans, x, kth, axis=-1, kind="introselect", order=None):
    # TODO: Cast input with np.asanyarray()
    if len(np.shape(x)) > 1:
        raise NotImplementedError(
            "Gradient of partition not implemented for multi-dimensional arrays."
        )
    partition_perm = np.argpartition(x, kth, axis, kind, order)
    return lambda g: unpermuter(g, partition_perm)


defvjp(np.partition, grad_partition)

# def unpermuter(g, permutation):
#     unsort = np.zeros(len(permutation), dtype=int)
#     unsort[permutation] = list(range(len(permutation)))
#     return g[unsort]

# def grad_reshape_list(ans, *arys):
#     if len(arys) > 1:
#         raise NotImplementedError("Can't handle multiple arguments yet.")
#     return lambda g: np.reshape(g, np.shape(arys[0]))
# defvjp(np.atleast_1d, grad_reshape_list)
# defvjp(np.atleast_2d, grad_reshape_list)
# defvjp(np.atleast_3d, grad_reshape_list)

# def grad_einsum(argnum, ans, operands_, kwargs):
#     result_meta = metadata(operands_[argnum])
#     def vjp(g):
#         operands = operands_
#         if isinstance(operands[0], string_types):  # using "ijk" convention.
#             in_subs, out_subs, _ = np.parse_einsum_input(*operands)
#             string, operands = operands[0], operands[1:]

#             in_subs_list = in_subs.split(',')
#             op_num = argnum - 1
#             subs_wrt = in_subs_list[op_num]
#             rest_of_ops = operands[:op_num] + operands[op_num+1:]
#             rest_of_subs = in_subs_list[:op_num] + in_subs_list[op_num+1:]

#             # subscripts that only appear in subs_wrt (and not in other subscript lists
#             # or in the output) are implicitly being summed out, as if contracted
#             # against a tensor of ones. we make that tensor of ones explicit to handle
#             # the necessary vjp broadcasting inside einsum.
#             other_named_subs = set(''.join([out_subs] + rest_of_subs))
#             naked_summed = [(i, sub) for i, sub in enumerate(subs_wrt)
#                             if sub not in other_named_subs]
#             if naked_summed:
#                 naked_summed_dims, ones_subs = zip(*naked_summed)
#                 ones_subs = ''.join(ones_subs)
#                 ones = onp.ones(onp.array(operands[op_num].shape)[list(naked_summed_dims)])
#                 new_input_subs = ','.join([out_subs, ones_subs] + rest_of_subs)
#                 new_operands = (g, ones) + rest_of_ops
#             else:
#                 new_input_subs = ','.join([out_subs] + rest_of_subs)
#                 new_operands = (g,) + rest_of_ops

#             new_subscripts = new_input_subs + '->' + subs_wrt
#             return unbroadcast(np.einsum(new_subscripts, *new_operands), result_meta)
#         else:  # using (op0, sublist0, op1, sublist1, ..., sublistout) convention
#             if len(operands) % 2 == 0:
#                 raise NotImplementedError("Need sublistout argument")
#             operands = list(operands)
#             rest_of_ops = [operands[-1]] + operands[:argnum] + \
#                     operands[(argnum+2):-1] + [operands[argnum+1]]
#             return unbroadcast_einsum(np.einsum(g, *rest_of_ops), result_meta, operands[argnum + 1])
#     return vjp
# defvjp_argnum(np.einsum, grad_einsum)

# defvjp(np.diagonal,
#     lambda ans, A, offset=0, axis1=0, axis2=1 :
#     lambda g: np.make_diagonal(g, offset, axis1, axis2))
# defvjp(np.make_diagonal,
#     lambda ans, D, offset=0, axis1=0, axis2=1 :
#     lambda g: np.diagonal(g, offset, axis1, axis2))


def match_complex(target, x):
    target_iscomplex = np.iscomplexobj(target)
    x_iscomplex = np.iscomplexobj(x)
    if x_iscomplex and not target_iscomplex:
        return np.real(x)
    elif not x_iscomplex and target_iscomplex:
        return x + 0j
    else:
        return x


def metadata(A):
    # return A.shape, A.ndim, A.dtype
    return np.shape(A), np.ndim(A), A.dtype


def unbroadcast(x, target_meta, broadcast_idx=0):
    target_shape, target_ndim, dtype = target_meta
    # while x.ndim > target_ndim:
    while np.ndim(x) > target_ndim:
        x = np.sum(x, axis=broadcast_idx)
    for axis, size in enumerate(target_shape):
        if size == 1:
            x = np.sum(x, axis=axis, keepdims=True)
    # if np.iscomplexobj(x) and not target_iscomplex:
    #     x = np.real(x)
    return x


def unbroadcast_f(target, f):
    target_meta = metadata(target)
    return lambda g: unbroadcast(f(g), target_meta)


# def unbroadcast_einsum(x, target_meta, subscript):
#     if Ellipsis not in subscript:
#         return x
#     elif subscript[0] == Ellipsis:
#         return unbroadcast(x, target_meta, 0)
#     elif subscript[-1] == Ellipsis:
#         return unbroadcast(x, target_meta, -1)
#     else:
#         return unbroadcast(x, target_meta, subscript.index(Ellipsis))


def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))


def replace_zero(x, val):
    return np.where(x, x, val)


def replace_non_positive(x, val):
    return np.where(x > 0, x, val)


# # ----- extra functions used internally  -----

# def array_from_args_gradmaker(argnum, ans, args, kwargs):
#     return lambda g: g[argnum-2]
# defvjp_argnum(np.array_from_args, array_from_args_gradmaker)

# def array_from_scalar_or_array_gradmaker(ans, array_args, array_kwargs, scarray):
#     ndmin = array_kwargs.get('ndmin', 0)
#     scarray_ndim = np.ndim(scarray)
#     if ndmin > scarray_ndim:
#         return lambda g: np.squeeze(g, axis=tuple(range(ndmin - scarray_ndim)))
#     else:
#         return lambda g: g
# defvjp(np._array_from_scalar_or_array, array_from_scalar_or_array_gradmaker, argnums=(2,3))

# @primitive
# def untake(x, idx, vs):
#     if isinstance(idx, list) and (len(idx) == 0 or not isinstance(idx[0], slice)):
#         idx = onp.array(idx, dtype='int64')
#     def mut_add(A):
#         onp.add.at(A, idx, x)
#         return A
#     return SparseObject(vs, mut_add)
# defvjp(func(ArrayBox.__getitem__), lambda ans, A, idx: lambda g: untake(g, idx, vspace(A)))
# defvjp(untake, lambda ans, x, idx, _: lambda g: g[idx])


def _unpad(array, width):
    if np.isscalar(width):
        width = [[width, width]]
    elif np.shape(width) == (1,):
        width = [np.concatenate((width, width))]
    elif np.shape(width) == (2,):
        width = [width]
    if np.shape(width)[0] == 1:
        width = np.repeat(width, np.ndim(array), 0)
    idxs = tuple(slice(l, -u or None) for l, u in width)
    return array[idxs]


def pad_vjp(ans, array, pad_width, mode, **kwargs):
    assert mode == "constant", "Only constant mode padding is supported."
    return lambda g: _unpad(g, pad_width)


defvjp(np.pad, pad_vjp)
