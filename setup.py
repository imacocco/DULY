from setuptools import setup, Extension


class get_numpy_include(object):
    """Defer numpy.get_include() until after numpy is installed.
    From: https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    """

    def __str__(self):
        import numpy

        return numpy.get_include()


ext_modules = []

ext_modules += [
    Extension(
        "dadapy._cython.cython_clustering",
        sources=["dadapy/_cython/cython_clustering.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_clustering_v2",
        sources=["dadapy/_cython/cython_clustering_v2.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_maximum_likelihood_opt",
        sources=["dadapy/_cython/cython_maximum_likelihood_opt.c"],
        include_dirs=[get_numpy_include()],
    )
]

ext_modules += [
    Extension(
        "dadapy._cython.cython_maximum_likelihood_opt_full",
        sources=["dadapy/_cython/cython_maximum_likelihood_opt_full.c"],
        include_dirs=[get_numpy_include()],
    )
]


ext_modules += [
    Extension(
        "dadapy._cython.cython_density",
        sources=["dadapy/_cython/cython_density.c"],
        include_dirs=[get_numpy_include()],
    )
]


setup(
    packages=["dadapy", "dadapy._utils"],
    ext_modules=ext_modules,
)
