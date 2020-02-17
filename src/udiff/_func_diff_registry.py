class FunctionDifferentialRegistry(dict):
    def __setitem__(self, k, v):
        if not callable(k):
            raise ValueError("key must be callable")

        if not callable(v):
            raise ValueError("value must be callable")

        super().__setitem__(k, v)


global_registry = FunctionDifferentialRegistry()


def diff(grad_f):
    def inner(f, registry=None):
        register_diff(f, grad_f, registry=registry)
        return f

    return inner


def register_diff(f, grad_f, registry=None):
    if registry is None:
        registry = global_registry
    registry[f] = grad_f
