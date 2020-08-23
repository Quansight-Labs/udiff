import uuid
import uarray as ua
import unumpy as np
import itertools
from ._core import primitive_vjps, primitive_jvps


class DiffArray(np.ndarray):
    def __init__(self, arr, mode):
        if isinstance(arr, DiffArray):
            self._value = arr.value
            self._id = arr.id
            self._parents = arr._parents
            self._jvp_diff = arr._jvp_diff
            self._vjp_func = arr._vjp_func
            self._vjp_diff = arr._vjp_diff
            self._jacobian = arr._jacobian
            self._mode = arr._mode
            return

        with ua.determine_backend(arr, np.ndarray, domain="numpy", coerce=True):
            arr = np.asarray(arr)

        self._value = arr
        self._mode = mode
        self._id = uuid.uuid4()
        self._parents = []
        self._jvp_diff = {}
        self._vjp_func = None
        self._vjp_diff = None
        self._jacobian = None

    @property
    def mode(self):
        return self._mode

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

        self._vjp_func = vjpmaker(tuple(parent_argnums), self, tuple(vjp_args), kwargs)

    def register_jvp(self, func, args, kwargs):
        try:
            if func is np.ufunc.__call__:
                jvpmaker = primitive_jvps[args[0]]
            else:
                jvpmaker = primitive_jvps[func]
        except KeyError:
            raise NotImplementedError("JVP of func not defined")

        pn = 0
        parent_argnums, jvp_args, start_nodes = [], [], []

        for arg in args:
            if isinstance(arg, DiffArray):
                self._parents.append((pn, arg))
                parent_argnums.append(pn)
                pn += 1
                jvp_args.append(arg.value)

                if not arg._jvp_diff:
                    arg._jvp_diff[arg] = np.ones_like(arg.value)

                start_nodes += list(arg._jvp_diff.keys())

            elif not isinstance(arg, np.ufunc):
                jvp_args.append(arg)

        for sn in set(start_nodes):
            parent_jvps = [
                parent._jvp_diff.get(sn, np.zeros_like(parent.value))
                for _, parent in self._parents
            ]

            self._jvp_diff[sn] = jvpmaker(
                tuple(parent_argnums), parent_jvps, self.value, tuple(jvp_args), kwargs
            )

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
            if self._vjp_diff is None:
                self._vjp_diff = {}

            if end_node in self._vjp_diff:
                self._vjp_diff[end_node] = self._vjp_diff[end_node] + grad_variables
            else:
                self._vjp_diff[end_node] = grad_variables

        if self._vjp_func:
            diffs = list(self._vjp_func(grad_variables))
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

        if self._vjp_func:
            diffs = list(self._vjp_func(grad_variables))
            for pn, p in self._parents:
                p._backward_jacobian(diffs[pn], end_node, position, base)

    def to(self, x, grad_variables=None, jacobian=False):
        if self.mode == "vjp":
            if jacobian:
                if x._jacobian is None or self not in x._jacobian:
                    for position in itertools.product(*[range(i) for i in self.shape]):
                        grad_variables = np.zeros_like(self.value)
                        grad_variables.value[position] = 1
                        self._backward_jacobian(grad_variables, self, position, base=x)

                x._jacobian[self] = np.reshape(
                    np.stack(x._jacobian[self].values()), self.shape + x.shape
                )
                return x._jacobian[self]
            else:
                if x._vjp_diff is None or self not in x._vjp_diff:
                    self.backward(grad_variables, base=x)
                return x._vjp_diff[self]
        elif self.mode == "jvp":
            if jacobian or grad_variables:
                raise NotImplementedError(
                    "JVP does not yet support grad_variables, jacobian and higher order derivative"
                )
            return self._jvp_diff[x]

    __repr__ = __str__
    __hash__ = object.__hash__
