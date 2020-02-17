import unumpy as np


class DiffArray(np.ndarray):
    def __init__(self, arr, diffs=None):
        if isinstance(arr, DiffArray):
            self._diffs = arr.diffs
            self._arr = arr.arr
            self._var = arr.var

        from udiff import SKIP_SELF
        with SKIP_SELF:
            arr = np.asarray(arr)

        if diffs is None:
            diffs = ArrayDiffRegistry(arr.shape)

        if not isinstance(diffs, ArrayDiffRegistry):
            raise ValueError("diffs must be an ArrayDiffRegistry")

        if diffs.shape != arr.shape:
            raise ValueError("diffs didn't have the right shape")

        self._diffs = diffs
        self._arr = arr
        self._var = None

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def diffs(self):
        return self._diffs

    @property
    def arr(self):
        return self._arr

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, value):
        if not isinstance(value, Variable):
            raise ValueError("var must be a Variable")

        if self._var is not None:
            raise ValueError("variable already set")

        self._var = value
        self.diffs[value] = np.array(1)

    def __getitem__(self, k):
        arr = self._arr[k]
        diffs = ArrayDiffRegistry(arr.shape)
        for var, darr in self.diffs.items():
            diffs[var] = darr[k]

        return DiffArray(arr, diffs=diffs)

    def __str__(self):
        return "<{}, name={}, arr=\n{}\n>".format(
            type(self).__name__,
            repr(self.var.name) if self.var is not None else "unbound",
            str(self.arr),
        )

    __repr__ = __str__
    __hash__ = object.__hash__

    def __setitem__(self, k, v):
        if not isinstance(v, type(self)):
            raise ValueError("v must be of the same type as self")

        if not all(i in v._diffs.keys() for i in self._diffs.keys()):
            raise ValueError("v doesn't have all required diffs")

        self._arr[k] = v._arr

        for i in self._diffs:
            self._diffs[i][k] = v._diffs[i]


class Variable:
    def __init__(self, name):
        if not isinstance(name, str):
            raise ValueError("k must be a string")

        self._name = str(name)

    @property
    def name(self):
        return self._name

    def __str__(self):
        return "Variable({})".format(repr(self._name))

    __repr__ = __str__


class ArrayDiffRegistry(dict):
    def __init__(self, shape):
        super().__init__()
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def __setitem__(self, k, v):
        if isinstance(k, DiffArray):
            if not k.var:
                raise ValueError("k does not have a set var")

            k = k.var

        if not isinstance(k, Variable):
            raise ValueError("k must be a Variable")

        if not isinstance(v, DiffArray):
            v = DiffArray(v)

        v = np.broadcast_to(v, self.shape)
        super().__setitem__(k, v)

    def __getitem__(self, k):
        if isinstance(k, DiffArray):
            if not k.var:
                raise ValueError("k does not have a set var")

            k = k.var
        return super().__getitem__(k)
