import unumpy as np
import uarray as ua
from unumpy import numpy_backend
import udiff


if __name__ == "__main__":
    with ua.set_backend(udiff.DiffArrayBackend(numpy_backend, mode="jvp"), coerce=True):
        u = np.array([[2, 2], [2, 2]])
        v = np.array([[5, 5], [5, 5]])
        y = np.log(u) + u * v - np.sin(v)
        print(y)
        print(y.to(u))
