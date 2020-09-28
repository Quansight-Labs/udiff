from collections import defaultdict
from itertools import count
from functools import reduce
import unumpy as np
import uarray as ua
from unumpy import numpy_backend

# -------------------- reverse mode --------------------

primitive_vjps = {}


def defvjp_argnums(fun, vjpmaker):
    primitive_vjps[fun] = vjpmaker


def defvjp_argnum(fun, vjpmaker):
    def vjp_argnums(argnums, *args):
        vjps = [vjpmaker(argnum, *args) for argnum in argnums]
        return lambda g: (vjp(g) for vjp in vjps)

    defvjp_argnums(fun, vjp_argnums)


def defvjp(fun, *vjpmakers, **kwargs):
    """
    Set up a unumpy-transformable function for a VJP rule definition.


    Parameters
    ----------
    fun : np.ufunc
        The function need to be derived.
    *jvpfuns :
        Functions for calculating derivative.

    Examples
    --------
    >>> defvjp(np.positive, lambda ans, x: lambda g: g)

    """
    argnums = kwargs.get("argnums", count())
    vjps_dict = {
        argnum: translate_vjp(vjpmaker, fun, argnum)
        for argnum, vjpmaker in zip(argnums, vjpmakers)
    }

    def vjp_argnums(argnums, ans, args, kwargs):
        try:
            vjps = [vjps_dict[argnum](ans, *args, **kwargs) for argnum in argnums]
        except KeyError:
            raise NotImplementedError("VJP of {} not defined".format(fun.name))

        def ret(g):
            return tuple(vjp(g) for vjp in vjps)

        return ret

    defvjp_argnums(fun, vjp_argnums)


def translate_vjp(vjpfun, fun, argnum):
    if vjpfun is None:
        return lambda ans, *args, **kwargs: lambda g: np.zeros_like(args[argnum])
    elif callable(vjpfun):
        return vjpfun
    else:
        raise Exception("Bad VJP '{}' for '{}'".format(vjpfun, fun.__name__))


# -------------------- forward mode --------------------

primitive_jvps = {}


def subval(x, i, v):
    x_ = list(x)
    x_[i] = v
    return tuple(x_)


def defjvp_argnums(fun, jvpmaker):
    primitive_jvps[fun] = jvpmaker


def defjvp_argnum(fun, jvpmaker):
    def jvp_argnums(argnums, ans, args, kwargs):
        return (jvpmaker(argnum, ans, args, kwargs) for argnum in argnums)

    defjvp_argnums(fun, jvp_argnums)


def defjvp(fun, *jvpfuns, **kwargs):
    """
    Set up a unumpy-transformable function for a JVP rule definition.

    Parameters
    ----------
    fun : np.ufunc
        The function need to be derived.
    *jvpfuns :
        Functions for calculating derivative.

    Examples
    --------
    >>> defjvp(
    ...     np.arctan2,
    ...     lambda ans, x, y: lambda g: g * y / (x ** 2 + y ** 2),
    ...     lambda ans, x, y: lambda g: g * -x / (x ** 2 + y ** 2),
    ... )

    """
    argnums = kwargs.get("argnums", count())
    jvps_dict = {
        argnum: translate_jvp(jvpfun, fun, argnum)
        for argnum, jvpfun in zip(argnums, jvpfuns)
    }

    def jvp_argnums(argnums, ans, args, kwargs):
        return [jvps_dict[argnum](ans, *args, **kwargs) for argnum in argnums]

    defjvp_argnums(fun, jvp_argnums)


def translate_jvp(jvpfun, fun, argnum):
    if jvpfun is None:
        return lambda ans, *a, **k: lambda g: np.zeros_like(ans)
    elif jvpfun == "same":
        return lambda ans, *args, **kwargs: lambda g: fun(
            *subval(args, argnum, g), **kwargs
        )
    elif callable(jvpfun):
        return jvpfun
    else:
        raise Exception("Bad JVP '{}' for '{}'".format(jvpfun, fun.__name__))


def def_linear(fun):
    """
    Flags that a function is linear wrt all args.

    Parameters
    ----------
    fun : np.ufunc
        The function need to be derived.

    Examples
    --------
    >>> def_linear(np.matmul)

    """
    defjvp_argnum(
        fun,
        lambda argnum, ans, args, kwargs: lambda g: fun(
            *subval(args, argnum, g), **kwargs
        ),
    )
