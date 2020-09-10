import uuid
import uarray as ua
import unumpy as np
import itertools
from functools import reduce
from ._core import primitive_vjps, primitive_jvps
from unumpy import numpy_backend


class DiffArray(np.ndarray):
    """
    A container with the necessary information used in derivation.

    Attributes
    ----------
    arr : DiffArray
        A DiffArray or ndarray used to initialize the class.


    .. note:: DiffArray is the base class of JVPDiffArray and VJPDiffArray. Do not use it.

    """

    def __init__(self, arr):
        if isinstance(arr, DiffArray):
            self._value = arr.value
            self._id = arr.id
            return

        with ua.determine_backend(arr, np.ndarray, domain="numpy", coerce=True):
            arr = np.asarray(arr)

        self._value = arr
        self._id = uuid.uuid4()

    @property
    def dtype(self):
        """
        The data type of the DiffArray.
        """
        return self._value.dtype

    @property
    def value(self):
        """
        The value of the DiffArray.
        """
        return self._value

    @property
    def id(self):
        """
        The id of the DiffArray.
        """
        return self._id

    def __str__(self):
        return "<{}, id={}, value=\n{}\n>".format(
            type(self).__name__, repr(self.id), str(self.value),
        )

    __repr__ = __str__
    __hash__ = object.__hash__


class VJPDiffArray(DiffArray):
    """
    A container with the necessary information used in derivation under VJP mode.

    Attributes
    ----------
    arr : VJPDiffArray or ndarray
        An VJPDiffArray or ndarray used to initialize the class.

    Examples
    --------
    You do not need to use VJPDiffArray explicitly.
    When you call a function such as ``np.array`` that could create a ndarray under vjp mode,
    VJPDiffArray will be created automatically.

    >>> with ua.set_backend(udiff.DiffArrayBackend(numpy_backend), coerce=True):
    ...    x = np.array([2])
    ...    isinstance(x, VJPDiffArray)
    True
    """

    def __init__(self, arr):
        if isinstance(arr, VJPDiffArray):
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
        self._parents = None
        self._vjp = None
        self._diff = None
        self._jacobian = None

    def register_diff(self, func, args, kwargs):
        """
        Register the derivative function used in backward propagation for the current node.

        Parameters
        ----------
        func : np.ufunc
            The function need to be derived.
        args :
            Arguments used in func.
        kwargs :
            Keyword-only arguments used in func.
        """

        try:
            if func is np.ufunc.__call__:
                vjpmaker = primitive_vjps[args[0]]
            else:
                vjpmaker = primitive_vjps[func]
        except KeyError:
            raise NotImplementedError("VJP of func not defined")

        vjp_args = []

        if self._parents is None:
            self._parents = []

        for arg in args:
            if isinstance(arg, VJPDiffArray):
                self._parents.append(arg)
                vjp_args.append(arg)
            elif not isinstance(arg, np.ufunc):
                vjp_args.append(arg)

        parent_argnums = tuple(range(len(self._parents)))
        self._vjp = vjpmaker(parent_argnums, self, tuple(vjp_args), kwargs)

    def _backward(self, grad_variables, end_node, base):
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
            for i, p in enumerate(self._parents):
                p._backward(diffs[i], end_node, base)

    def _backward_jacobian(self, grad_variables, end_node, position, base):
        if base is None or base.id == self.id:
            if self._jacobian is None:
                self._jacobian = {}

            if end_node not in self._jacobian:
                self._jacobian[end_node] = {}

            if position not in self._jacobian[end_node]:
                self._jacobian[end_node][position] = grad_variables
            else:
                self._jacobian[end_node][position] = (
                    self._jacobian[end_node][position] + grad_variables
                )

        if self._vjp:
            diffs = list(self._vjp(grad_variables))
            for i, p in enumerate(self._parents):
                p._backward_jacobian(diffs[i], end_node, position, base)

    def to(self, x, grad_variables=None, jacobian=False):
        """
        Calculate the VJP or Jacobian matrix of self to x.

        Parameters
        ----------
        x : VJPDiffArray
            The denominator in derivative.
        grad_variables : VJPDiffArray
            Gradient of the numerator in derivative.
        jacobian : bool
            Flag identifies whether to calculate the jacobian logo.
            If set ``True``, it will return jacobian matrix instead of vjp.

        Examples
        --------
        >>> with ua.set_backend(udiff.DiffArrayBackend(numpy_backend), coerce=True):
        ...
        ...    x1 = np.array([2])
        ...    x2 = np.array([5])
        ...    y = np.log(x1) + x1 * x2 - np.sin(x2)
        ...    x1_diff = y.to(x1)
        ...    print(np.allclose(x1_diff.value, [5.5]))
        True
        """
        if jacobian:
            if x._jacobian is None or self not in x._jacobian:
                for position in itertools.product(*[range(i) for i in np.shape(self)]):
                    grad_variables = np.zeros_like(self.value)
                    grad_variables.value[position] = 1
                    self._backward_jacobian(grad_variables, self, position, x)

            x._jacobian[self] = np.reshape(
                np.stack(x._jacobian[self].values()), np.shape(self) + np.shape(x)
            )
            return x._jacobian[self]
        else:
            if x._diff is None or self not in x._diff:
                self._backward(grad_variables, self, x)
            return x._diff[self]


class JVPDiffArray(DiffArray):
    """
    A container with the necessary information used in derivation under jvp mode.

    Attributes
    ----------
    arr : JVPDiffArray
        A JVPDiffArray or ndarray used to initialize the class.

    Examples
    --------
    You do not need to use JVPDiffArray explicitly.
    When you call a function such as ``np.array`` that could create a ndarray under jvp mode,
    JVPDiffArray will be created automatically.

    >>> with ua.set_backend(udiff.DiffArrayBackend(numpy_backend, mode="jvp"), coerce=True):
    ...    x = np.array([2])
    ...    isinstance(x, JVPDiffArray)
    True
    """

    def __init__(self, arr):
        if isinstance(arr, JVPDiffArray):
            self._value = arr.value
            self._id = arr.id
            self._diff = arr._diff
            self._jacobian = arr._jacobian
            self._jvp = arr._jvp
            self._vars = arr._vars
            return

        with ua.determine_backend(arr, np.ndarray, domain="numpy", coerce=True):
            arr = np.asarray(arr)

        self._value = arr
        self._id = uuid.uuid4()
        self._diff = None
        self._jvp = None
        self._jacobian = None
        self._vars = None

    def register_diff(self, func, args, kwargs):
        """
        Calculate the derivative of the current node to the previous node.

        Parameters
        ----------
        func : np.ufunc
            The function need to be derived.
        args :
            Arguments used in func.
        kwargs :
            Keyword-only arguments used in func.
        """
        try:
            if func is np.ufunc.__call__:
                jvpmaker = primitive_jvps[args[0]]
            else:
                jvpmaker = primitive_jvps[func]
        except KeyError:
            raise NotImplementedError("JVP of func not defined")

        jvp_args, parents = [], []

        for arg in args:
            if isinstance(arg, JVPDiffArray):
                jvp_args.append(arg)

                parents.append(arg)

            elif not isinstance(arg, np.ufunc):
                jvp_args.append(arg)

        parent_argnums = tuple(range(len(parents)))

        jvps = list(jvpmaker(parent_argnums, self, tuple(jvp_args), kwargs))

        if self._jvp is None:
            self._jvp = {}

        for p, jvp in zip(parents, jvps):
            if p not in self._jvp:
                self._jvp[p] = [[jvp]]
            else:
                self._jvp[p] += [[jvp]]
            if p._jvp:
                for base in p._jvp:
                    if base not in self._jvp:
                        self._jvp[base] = []
                    self._jvp[base] += [flist + [jvp] for flist in p._jvp[base]]

    def _forward(self, x, grad_variables):
        if self._jvp is None:
            return grad_variables

        result = []

        for flist in self._jvp[x]:
            cur_result = grad_variables
            for f in flist:
                cur_result = f(cur_result)
            result.append(cur_result)

        return reduce(lambda x, y: x + y, result)

    def to(self, x, grad_variables=None, jacobian=False):
        """
        Calculate the JVP or Jacobian matrix of self to x.

        .. note::  JVP does not yet support higher order derivative.

        Parameters
        ----------
        x : JVPDiffArray
            The denominator in derivative.
        grad_variables : JVPDiffArray
            Gradient assigned to the x.
        jacobian : bool
            Flag identifies whether to calculate the jacobian logo.
            If set ``True``, it will return jacobian matrix instead of jvp.

        Examples
        --------
        >>> with ua.set_backend(udiff.DiffArrayBackend(numpy_backend, mode="jvp"), coerce=True):
        ...
        ...    x1 = np.array([2])
        ...    x2 = np.array([5])
        ...    y = np.log(x1) + x1 * x2 - np.sin(x2)
        ...    x1_diff = y.to(x1)
        ...    print(np.allclose(x1_diff, [5.5]))
        True
        """
        if self._jvp and x not in self._jvp:
            raise ValueError("Please check if the base is correct.")

        if jacobian:
            if self._jacobian is None:
                self._jacobian = {}

            if x not in self._jacobian:
                self._jacobian[x] = {}
                for position in itertools.product(*[range(i) for i in np.shape(x)]):
                    grad_variables = np.zeros_like(x)
                    grad_variables.value[position] = 1
                    self._jacobian[x][position] = self._forward(x, grad_variables)

            old_axes = tuple(range(np.ndim(self) + np.ndim(x)))
            new_axes = old_axes[np.ndim(x) :] + old_axes[: np.ndim(x)]
            self._jacobian[x] = np.transpose(
                np.reshape(
                    np.stack(self._jacobian[x].values()), np.shape(x) + np.shape(self),
                ),
                new_axes,
            )
            return self._jacobian[x]
        else:
            if self._diff is None:
                self._diff = {}

            if x not in self._diff:
                if grad_variables is None:
                    grad_variables = np.ones_like(self)

                self._diff[x] = self._forward(x, grad_variables)

            return self._diff[x]
