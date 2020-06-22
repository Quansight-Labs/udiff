from uarray import wrap_single_convertor
from unumpy import ufunc, ndarray
import unumpy

import unumpy as np
import uarray as ua

from ._diff_array import DiffArray
from ._vjp_diffs import nograd_functions

from typing import Dict

_ufunc_mapping: Dict[ufunc, np.ufunc] = {}

__ua_domain__ = "numpy"

_implementations: Dict = {
    unumpy.arange: lambda start, stop=None, step=None, **kw: da.arange(
        start, stop, step, **kw
    ),
    unumpy.asarray: DiffArray,
}


def __ua_function__(func, args, kwargs, requires_grad=True):
    from udiff import SKIP_SELF

    extracted_args = func.arg_extractor(*args, **kwargs)
    arr_args = tuple(x.value for x in extracted_args if x.type is np.ndarray)

    with SKIP_SELF:
        if len(arr_args) == 0:
            out = func(*args, **kwargs)
            # return DiffArray(out)
        else:
            a, kw = replace_arrays(
                func,
                args,
                kwargs,
                (
                    x.value if x is not None and isinstance(x, DiffArray) else x
                    for x in arr_args
                ),
            )
            out = func(*a, **kw)

    if requires_grad:
        out = DiffArray(out)

        if func not in nograd_functions:
            with ua.set_backend(NoGradBackend()):
                out.register_vjp(func, args, kwargs)

    return out


def replace_arrays(func, a, kw, arrays):
    d = tuple(func.arg_extractor(*a, **kw))
    arrays = tuple(arrays)
    new_d = []
    j = 0
    for i in d:
        if i.type is np.ndarray:
            new_d.append(arrays[j])
            j += 1
        else:
            new_d.append(i.value)

    return func.arg_replacer(a, kw, tuple(new_d))


@wrap_single_convertor
def __ua_convert__(value, dispatch_type, coerce):
    if dispatch_type is np.ndarray:
        if value is None:
            return value

        if isinstance(value, DiffArray):
            return value

        if coerce:
            import udiff

            with udiff.SKIP_SELF:
                return DiffArray(np.asarray(value))

        return NotImplemented

    return value


class NoGradBackend:
    __ua_domain__ = __ua_domain__
    __ua_convert__ = staticmethod(__ua_convert__)

    def __ua_function__(self, f, a, kw):

        return __ua_function__(f, a, kw, False)

    def __eq__(self, other):
        import udiff

        return isinstance(other, NoGradBackend) or other is udiff
