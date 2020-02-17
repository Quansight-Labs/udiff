# Contributing

Contributions to `udiff` are welcome and appreciated. Contributions can take the form of bug reports, documentation, code, and more.

## Getting the code

Make a fork of the main [udiff repository](https://github.com/Quansight-Labs/udiff) and clone the fork:

```
git clone https://github.com/<your-github-username>/udiff
```

## Install

Note that udiff supports Python versions >= 3.5. If you're running `conda` and would prefer to have dependencies
pulled from there, use

```
conda env create -f .conda/environment.yml
conda activate uarray
```

This will create an environment named `uarray` which you can use for development.

`unumpy` and all development dependencies can be installed via:

```
pip install -e .
```

## Testing

Tests can be run from the main uarray directory as follows:

```
pytest
```
