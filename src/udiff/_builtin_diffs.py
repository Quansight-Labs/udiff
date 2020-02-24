import unumpy as np

from ._func_diff_registry import register_diff


def multiply_diff(x1, x2):
    return x1[0] * x2[1] + x1[1] * x2[0]


def divide_diff(x1, x2):
    fprimeg = np.multiply(x1[1], x2[0])
    gprimef = np.multiply(x1[0], x2[1])
    gsquared = np.multiply(x2[0], x2[0])
    return (fprimeg - gprimef) / gsquared


def pow_diff(x1, x2):
    ftog = x1[0] ** x2[0]
    gplogf = np.log(x1[0]) * x2[1]
    gfpof = x1[1] / x1[0] * x2[0]
    return ftog * (gplogf + gfpof)


def matmul_diff(x1, x2):
    return x1[0] @ x2[1] + x1[1] @ x2[0]


def arctan2_diff(x1, x2):
    return (x1[1] * x2[0] - x1[0] * x2[1]) / (np.square(x1[0]) + np.square(x2[0]))


register_diff(np.sign, lambda x: np.broadcast_to(np.where(x[0].arr == 0, float('nan'), 0), x[1].shape))
register_diff(np.add, lambda x1, x2: x1[1] + x2[1])
register_diff(np.subtract, lambda x1, x2: x1[1] - x2[1])
register_diff(np.multiply, multiply_diff)
register_diff(np.matmul, matmul_diff)
register_diff(np.divide, divide_diff)
register_diff(np.true_divide, divide_diff)
register_diff(np.power, pow_diff)
register_diff(np.absolute, lambda x: x[1] * np.where(np.sign(x[0]) == 0, float('nan'), np.sign(x[0])))
register_diff(np.positive, lambda x: +x[1])
register_diff(np.negative, lambda x: -x[1])
register_diff(np.conj, lambda x: np.conj(x[1]))
register_diff(np.exp, lambda x: x[1] * np.exp(x[0]))
register_diff(np.exp2, lambda x: x[1] * np.log(2) * np.exp2(x[0]))
register_diff(np.log, lambda x: x[1] / x[0])
register_diff(np.log2, lambda x: x[1] / (np.log(2) * x[0]))
register_diff(np.log10, lambda x: x[1] / (np.log(10) * x[0]))
register_diff(np.sqrt, lambda x: x[1] / (2 * np.sqrt(x[0])))
register_diff(np.square, lambda x: 2 * x[1] * x[0])
register_diff(np.cbrt, lambda x: x[1] / (3 * (x[0] ** (2 / 3))))
register_diff(np.reciprocal, lambda x: -x[1] / np.square(x[0]))
register_diff(np.broadcast_to, lambda x, shape: np.broadcast_to(x[1], shape))

register_diff(np.sin, lambda x: x[1] * np.cos(x[0]))
register_diff(np.cos, lambda x: -x[1] * np.sin(x[0]))
register_diff(np.tan, lambda x: x[1] / np.square(np.cos(x[0])))
register_diff(np.arcsin, lambda x: x[1] / np.sqrt(1 - np.square(x[0])))
register_diff(np.arccos, lambda x: -x[1] / np.sqrt(1 - np.square(x[0])))
register_diff(np.arctan, lambda x: x[1] / (1 + np.square(x[0])))
register_diff(np.arctan2, arctan2_diff)

register_diff(np.sinh, lambda x: x[1] * np.cosh(x[0]))
register_diff(np.cosh, lambda x: x[1] * np.sinh(x[0]))
register_diff(np.tanh, lambda x: x[1] / np.square(np.cosh(x[0])))
register_diff(np.arcsinh, lambda x: x[1] / np.sqrt(1 + np.square(x[0])))
register_diff(np.arccosh, lambda x: x[1] / np.sqrt(1 - np.square(x[0])))
register_diff(np.arctanh, lambda x: x[1] / (1 - np.square(x[0])))
