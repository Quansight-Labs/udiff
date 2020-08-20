import uuid
import uarray as ua
import unumpy as np
import itertools
from ._vjp_core import primitive_vjps

import unumpy as np


class DiffArray(np.ndarray):
    def __init__(self, arr):
        if isinstance(arr, DiffArray):
            self._value = arr.value
            self._id = arr.id
            self._parents = arr._parents
            self._vjp = arr._vjp
            self._diff = arr._diff
            self._jacobian = arr._jacobian
            return

        with ua.determine_backend(arr, np.ndarray, domain="numpy", coerce=True):
            arr = np.asarray(arr)

        self._value = arr
        self._id = uuid.uuid4()
        self._parents = []
        self._vjp = None
        self._diff = None
        self._jacobian = None

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def shape(self):
        return np.shape(self.value)

    @property
    def ndim(self):
        return np.shape(self.value)

    @property
    def value(self):
        return self._value

    @property
    def id(self):
        return self._id

    @property
    def diff(self):
        return self._diff

    @property
    def jacobian(self):
        return self._jacobian

    def register_vjp(self, func, args, kwargs):
        try:
            if func is np.ufunc.__call__:
                vjpmaker = primitive_vjps[args[0]]
            else:
                vjpmaker = primitive_vjps[func]
        except KeyError:
            raise NotImplementedError("VJP of func not defined")

        parent_argnums, vjp_args = [], []
        pn = 0

        for arg in args:
            if isinstance(arg, DiffArray):
                self._parents.append((pn, arg))
                parent_argnums.append(pn)
                pn += 1
                vjp_args.append(arg)
            elif not isinstance(arg, np.ufunc):
                vjp_args.append(arg)

        self._vjp = vjpmaker(tuple(parent_argnums), self, tuple(vjp_args), kwargs)

    def __str__(self):
        return "<{}, id={}, value=\n{}\n>".format(
            type(self).__name__, repr(self.id), str(self.value),
        )

    def backward(self, grad_variables=None, end_node=None, base=None):
        """
        Backpropagation.
        Traverse computation graph backwards in topological order from the end node.
        For each node, compute local gradient contribution and accumulate.
        """
        if grad_variables is None:
            grad_variables = np.ones_like(self.value)

        if end_node is None:
            end_node = self

        if base is None or base.id == self.id:
            if self._diff is None:
                self._diff = {}

            if end_node in self._diff:
                self._diff[end_node] = self._diff[end_node] + grad_variables
            else:
                self._diff[end_node] = grad_variables

        if self._vjp:
            diffs = list(self._vjp(grad_variables))
            for pn, p in self._parents:
                p.backward(diffs[pn], end_node, base)

    def _backward_jacobian(self, grad_variables, end_node, position, base):
        if base is None or base.id == self.id:
            if self._jacobian is None:
                self._jacobian = {}

            if end_node not in self._jacobian:
                self._jacobian[end_node] = {}

            if position not in self._jacobian:
                self._jacobian[end_node][position] = grad_variables
            else:
                self._jacobian[end_node][position] = (
                    self._jacobian[end_node][position] + grad_variables
                )

        if self._vjp:
            diffs = list(self._vjp(grad_variables))
            for pn, p in self._parents:
                p._backward_jacobian(diffs[pn], end_node, position, base)

    def to(self, x, grad_variables=None, jacobian=False):
        if jacobian:
            if x._jacobian is None or self not in x._jacobian:
                for position in itertools.product(*[range(i) for i in self.shape]):
                    grad_variables = np.zeros_like(self.value)
                    grad_variables._value[position] = 1
                    self._backward_jacobian(grad_variables, self, position, base=x)

            x.jacobian[self] = np.reshape(
                np.stack(x.jacobian[self].values()), self.shape + x.shape
            )
            return x.jacobian[self]
        else:
            if x._diff is None or self not in x._diff:
                self.backward(grad_variables, base=x)
            return x._diff[self]

    __repr__ = __str__
    __hash__ = object.__hash__
