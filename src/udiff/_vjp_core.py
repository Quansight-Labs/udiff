from collections import defaultdict
from itertools import count
import unumpy as np
import uarray as ua


primitive_vjps = {}


def defvjp_argnums(fun, vjpmaker):
    primitive_vjps[fun] = vjpmaker


def defvjp_argnum(fun, vjpmaker):
    def vjp_argnums(argnums, *args):
        vjps = [vjpmaker(argnum, *args) for argnum in argnums]
        return lambda g: (vjp(g) for vjp in vjps)

    defvjp_argnums(fun, vjp_argnums)


def defvjp(fun, *vjpmakers, **kwargs):
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
