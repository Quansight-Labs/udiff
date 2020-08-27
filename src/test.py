import uarray as ua
import unumpy as np
from unumpy import numpy_backend
import udiff

if __name__ == "__main__":
    with ua.set_backend(udiff.DiffArrayBackend(numpy_backend, mode="jvp"), coerce=True):
        u = np.array([[2, 2], [2, 2]])
        v = np.array([[5, 5], [5, 5]])
        with ua.set_backend(numpy_backend, coerce=True):
            grad_variables = np.array([[0, 1], [1, 0]])
            u.set_grad_variables(grad_variables)
        y = np.log(u) + u * v - np.sin(v)
        print(y)
        print(y.to(u))
        print(y.to(v))
