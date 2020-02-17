from uarray import wrap_single_convertor
from unumpy import ufunc, ndarray
import unumpy
import functools
import unumpy as np
import uarray as ua
from . import _builtin_diffs

from ._diff_array import DiffArray

from typing import Dict

_ufunc_mapping: Dict[ufunc, np.ufunc] = {}

__ua_domain__ = "numpy"

_implementations: Dict = {
    unumpy.arange: lambda start, stop=None, step=None, **kw: da.arange(
        start, stop, step, **kw
    ),
    unumpy.asarray: DiffArray,
}


def __ua_function__(func, args, kwargs, tree=None):
    from udiff import SKIP_SELF
    from ._func_diff_registry import global_registry

    extracted_args = func.arg_extractor(*args, **kwargs)
    arr_args = tuple(x.value for x in extracted_args if x.type is np.ndarray)
    input_args = tuple(
        x.value for x in extracted_args if x.coercible and x.type is np.ndarray
    )

    if tree is None:
        tree = compute_diff_tree(*input_args)

    with SKIP_SELF:
        if len(arr_args) == 0:
            out = func(*args, **kwargs)
            return DiffArray(out)

        a, kw = replace_arrays(
            func, args, kwargs, (x.arr if x is not None else None for x in arr_args)
        )
        out_arr = func(*a, **kw)

    out = DiffArray(out_arr)
    for k in tree:
        diff_args = []
        for arr in arr_args:
            if arr is None:
                diff_args.append(None)
                continue

            if k in arr.diffs:
                diff_args.append((arr, arr.diffs[k]))
            else:
                diff_args.append((arr, np.broadcast_to(0, arr.shape)))

        a, kw = replace_arrays(func, args, kwargs, diff_args)

        with ua.set_backend(NoRecurseBackend(tree[k])):
            if func is np.ufunc.__call__:
                diff_arr = global_registry[a[0]](*a[1:], **kw)
            else:
                diff_arr = global_registry[func](*a, **kw)
            out.diffs[k] = diff_arr

    return out


def compute_diff_tree(*arrs, diffs=None):
    if diffs is None:
        diffs = {}
    for arr in arrs:
        for var, diff in arr.diffs.items():
            diffs[var] = compute_diff_tree(diff, diffs=diffs.get(var, {}))
    return diffs


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


def replace_self(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        if self not in _ufunc_mapping:
            return NotImplemented

        return func(_ufunc_mapping[self], *args, **kwargs)

    return inner


class NoRecurseBackend:
    def __init__(self, tree=None):
        self._tree = tree

    __ua_domain__ = __ua_domain__
    __ua_convert__ = staticmethod(__ua_convert__)

    def __ua_function__(self, f, a, kw):
        return __ua_function__(f, a, kw, tree=self._tree)

    def __eq__(self, other):
        import udiff
        return isinstance(other, NoRecurseBackend) or other is udiff
