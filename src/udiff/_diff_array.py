import uarray as ua
import unumpy as np
import numpy as onp
from ._vjp_core import primitive_vjps

import unumpy as np


class DiffArray(np.ndarray):
    count = 0

    def __init__(self, arr):
        if isinstance(arr, DiffArray):
            self._value = arr.value
            self._name = arr.name
            self._parents = arr.parents
            self._vjp = arr.vjp
            self._diff = arr._diff
            self._jacobian = arr._jacobian

        from udiff import SKIP_SELF

        with SKIP_SELF:
            arr = np.asarray(arr)

        self._value = arr
        self._name = "var_{}".format(DiffArray.count)
        self._parents = []
        self._vjp = None
        self._diff = None
        self._jacobian = None
        self._visit = None

        DiffArray.count += 1

    @property
    def shape(self):
        return self._value.shape

    @property
    def ndim(self):
        return self._value.ndim

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def value(self):
        return self._value

    @property
    def name(self):
        return self._name

    @property
    def diff(self):
        return self._diff

    def register_vjp(self, func, args, kwargs):
        try:
            if func is np.ufunc.__call__:
                vjpmaker = primitive_vjps[args[0]]
            else:
                vjpmaker = primitive_vjps[func]
        except KeyError:
            raise NotImplementedError("VJP of func not defined")
        parent_argnums = []
        vjp_args = []
        pn = 0
        for arg in args:
            if isinstance(arg, DiffArray):
                self._parents.append((pn, arg))
                parent_argnums.append(pn)
                pn += 1
                vjp_args.append(arg.value)

        self._vjp = vjpmaker(tuple(parent_argnums), self.value, tuple(vjp_args), kwargs)

    def __str__(self):
        return "<{}, name={}, value=\n{}\n>".format(
            type(self).__name__,
            repr(self.name) if self.name is not None else "unbound",
            str(self.value),
        )

    def backward(self, grad_variables=None, end_node=None):
        """
        Backpropagation.
        Traverse computation graph backwards in topological order from the end node.
        For each node, compute local gradient contribution and accumulate.
        """
        from ._uarray_plug import NoGradBackend

        with ua.set_backend(NoGradBackend()):
            if grad_variables is None:
                grad_variables = np.ones(self.value.shape)
            if end_node is None:
                end_node = self.name
            if self._diff is None or self._visit != end_node:
                self._diff = np.zeros(self.value.shape)
            self._diff += grad_variables
            self._visit = end_node
            if self._vjp:
                diffs = list(self._vjp(grad_variables))
                for pn, p in self._parents:
                    p.backward(diffs[pn], self._visit)

    def backward_jacobian(self):
        """Backpropagation.
        Traverse computation graph backwards in topological order from the end node.
        For each node, compute local Jacobian contribution and accumulate.
        """
        raise NotImplementedError

    __repr__ = __str__
    __hash__ = object.__hash__
