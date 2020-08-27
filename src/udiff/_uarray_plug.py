import functools

from uarray import wrap_single_convertor_instance
from unumpy import ufunc, ndarray, numpy_backend
import unumpy

import unumpy as np
import uarray as ua

from ._diff_array import DiffArray, VJPDiffArray, JVPDiffArray
from ._vjp_diffs import nograd_functions, raw_functions

from typing import Dict

_ufunc_mapping: Dict[ufunc, np.ufunc] = {}


class DiffArrayBackend:
    """
    The backend used for udiff.

    Attributes
    ----------
    _inner
        The backend used, such as numpy_backend.

    _mode: default vjp.
        The mode used to calculate gradient. It must be vjp or jvp.

    Examples
    --------
    >>> with ua.set_backend(udiff.DiffArrayBackend(numpy_backend), coerce=True):
    ...    x = np.array([2])
    """

    __ua_domain__ = "numpy"

    _implementations: Dict = {
        unumpy.asarray: DiffArray,
    }

    @property
    @functools.lru_cache(None)
    def self_implementations(self):
        """
        Specify the data type to be converted.
        """
        return {unumpy.ClassOverrideMeta.overridden_class.fget: self.overridden_class}

    def __init__(self, inner, mode="vjp"):
        mode = mode.lower()
        if mode not in ["vjp", "jvp"]:
            raise ValueError("mode must be vjp or jvp")
        self._inner = inner
        self._mode = mode

    def overridden_class(self, self2):
        """
        Convert ndarray to VJPDiffArray or JVPDiffArray according to mode.
        """
        if self is ndarray:
            if self._mode == "vjp":
                return VJPDiffArray
            else:
                return JVPDiffArray

        with ua.set_backend(self._inner, only=True):
            return self2.overridden_class

    def __ua_function__(self, func, args, kwargs):
        extracted_args = func.arg_extractor(*args, **kwargs)
        arr_args = tuple(x.value for x in extracted_args if x.type is np.ndarray)

        with ua.set_backend(self._inner, only=True):
            if len(arr_args) == 0:
                out = func(*args, **kwargs)
            else:
                a, kw = self.replace_arrays(
                    func,
                    args,
                    kwargs,
                    (
                        x.value if x is not None and isinstance(x, DiffArray) else x
                        for x in arr_args
                    ),
                )
                out = func(*a, **kw)

        real_func = func
        if func is np.ufunc.__call__:
            real_func = args[0]

        if real_func not in raw_functions:
            with ua.set_backend(self._inner, coerce=True):
                if self._mode == "vjp":
                    out = VJPDiffArray(out)
                else:
                    out = JVPDiffArray(out)

            if real_func not in nograd_functions:
                out.register_diff(func, args, kwargs)

        return out

    def replace_arrays(self, func, a, kw, arrays):
        """
        Convert the parameters in func to primitive types.
        """
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

    @wrap_single_convertor_instance
    def __ua_convert__(self, value, dispatch_type, coerce):
        if dispatch_type is np.ndarray:
            if value is None:
                return value

            if isinstance(value, DiffArray):
                return value

            if coerce:
                with ua.set_backend(self._inner, coerce=True):
                    if self._mode == "vjp":
                        return VJPDiffArray(np.asarray(value))
                    else:
                        return JVPDiffArray(np.asarray(value))

            return NotImplemented

        return value

    __hash__ = object.__hash__
    __eq__ = object.__eq__
