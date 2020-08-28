import sys
import uarray
import unumpy
import numpy
from unumpy import numpy_backend
import udiff
import pytest  # type: ignore


def pytest_cmdline_preparse(args):
    try:
        import pytest_black  # type: ignore
    except ImportError:
        pass
    else:
        args.append("--black")
        print("uarray: Enabling pytest-black")


@pytest.fixture(autouse=True)
def add_namespaces(doctest_namespace):
    doctest_namespace["ua"] = uarray
    doctest_namespace["np"] = unumpy
    doctest_namespace["onp"] = numpy
    doctest_namespace["udiff"] = udiff
    doctest_namespace["numpy_backend"] = numpy_backend
    doctest_namespace["broadcast"] = udiff._jvp_diffs.broadcast
