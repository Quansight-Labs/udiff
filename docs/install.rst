Prerequisites
--------------

There some prerequisites packages you should install before udiff.

:Python3: udiff requires Python 3.5 or higher.
:`uarray <https://github.com/Quansight-Labs/uarray>`_: uarray is a backend system for Python that allows you to separately define an API,
 along with backends that contain separate implementations of that API.

    .. code:: bash

        pip install git+https://github.com/Quansight-Labs/uarray.git

:`unumpy <https://github.com/Quansight-Labs/unumpy>`_: unumpy builds on top of uarray.
 It is an effort to specify the core NumPy API, and provide backends for the API.

    .. code:: bash

        pip install git+https://github.com/Quansight-Labs/unumpy.git


Installation
-------------

.. note::
    :obj:`udiff` has not been published on PyPI. You have to install it from source code now.

#.  Use Git to clone the :obj:`udiff` repository:

    .. code:: bash

        git clone https://github.com/Quansight-Labs/udiff.git
        cd udiff

#.  Install :obj:`udiff` on the command line, enter:

    .. code:: bash

        pip install -e . --no-deps --user

    If you want to install it system-wide for all users (assuming you have the necessary rights),
    just drop the ``--user`` flag.
