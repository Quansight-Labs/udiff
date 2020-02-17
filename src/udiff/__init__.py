import sys
import uarray as ua


from ._uarray_plug import __ua_domain__, __ua_convert__, __ua_function__
from ._func_diff_registry import (
    FunctionDifferentialRegistry,
    register_diff,
    diff,
)
from ._diff_array import Variable, DiffArray, ArrayDiffRegistry

__all__ = [
    "__ua_domain__",
    "__ua_convert__",
    "__ua_function__",
    "FunctionDifferentialRegistry",
    "register_diff",
    "diff",
    "Variable",
    "DiffArray",
    "ArrayDiffRegistry",
]

SKIP_SELF = ua.skip_backend(sys.modules["udiff"])
