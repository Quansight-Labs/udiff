import setuptools

setuptools.setup(
    name="udiff",
    version="0.6.0",
    author="Hameer Abbasi",
    author_email="hameerabbasi@yahoo.com",
    description="Automatic differentiation with uarray/unumpy.",
    platforms="Posix; MacOS X; Windows",
    packages=setuptools.find_packages(where="src", exclude=["tests*"]),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=("uarray >= 0.6.0", "unumpy >= 0.6.0"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    project_urls={"Source": "https://github.com/Quansight-Labs/udiff",},
    zip_safe=False,
)
