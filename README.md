<img src="https://raw.githubusercontent.com/sissa-data-science/DADApy/master/logo/logo_1_horizontal_transparent_v2.png" width="500">

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![codecov](https://codecov.io/gh/sissa-data-science/DADApy/branch/main/graph/badge.svg?token=X4M0KWAPO5)](https://codecov.io/gh/sissa-data-science/DADApy)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sissa-data-science/dadapy/test.yml?label=test)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sissa-data-science/dadapy/lint.yml?label=lint)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sissa-data-science/dadapy/lint.yml?label=docs)

DADApy is a Python package for the characterization of manifolds in high-dimensional spaces.


# Homepage
For more details and tutorials, visit the homepage at:
[https://dadapy.readthedocs.io/](https://dadapy.readthedocs.io/)

# Quick Example

```python
import numpy as np
from dadapy.data import Data

# Generate a simple 3D gaussian dataset
X = np.random.normal(0, 1, (1000, 3))

# initialize the "Data" class with the set of coordinates
data = Data(X)

# compute distances up to the 100th nearest neighbor
data.compute_distances(maxk=100)

# compute the intrinsic dimension using 2nn estimator
id, id_error, id_distance = data.compute_id_2NN()

# compute the intrinsic dimension up to the 64th nearest neighbors using Gride
id_list, id_error_list, id_distance_list = data.return_id_scaling_gride(range_max=64)

# compute the density using PAk, a point adaptive kNN estimator
log_den, log_den_error = data.compute_density_PAk()

# find the peaks of the density profile through the ADP algorithm
cluster_assignment = data.compute_clustering_ADP()

# compute the neighborhood overlap with another dataset
X2 = np.random.normal(0, 1, (1000, 5))
overlap_x2 = data.return_data_overlap(X2)

# compute the neighborhood overlap with a set of labels
labels = np.repeat(np.arange(10), 100)
overlap_labels = data.return_label_overlap(labels)

```

# Currently implemented algorithms

- Intrinsic dimension estimators
     - 
- Two-NN estimator 
  > Facco et al., *Scientific Reports* (2017)
- Gride estimator
  > Denti et al., *Scientific Reports* (2022)
- I3D estimator (for both continuous and discrete spaces)
  > Macocco et al., *Physical Review Letters* (2023)
- Density estimators
    - 
- kNN estimator
- k*NN estimator (kNN with an adaptive choice of k)
- PAk estimator
  > Rodriguez et al., *JCTC* (2018)
- BMTI estimator
  > Carli et al., *in preparation*

- Density peaks clustering methods
    - 
- Density peaks clustering 
  > Rodriguez and Laio, *Science* (2014)
- Advanced density peaks clustering
  > d’Errico et al., *Information Sciences* (2021)
- k-peak clustering
  > Sormani, Rodriguez and Laio, *JCTC* (2020)

- Manifold comparison tools
    - 
- Neighbourhood overlap
  > Doimo et al., *NeurIPS* (2020)
- Information imbalance
  > Glielmo et al., *PNAS Nexus* (2022)

- Feature selection and weighting tool
    -
- Differentiable Information Imbalance


# Installation
The package is compatible with the Python versions 3.7, 3.8, 3.9, 3.10, 3.11, and 3.12. We currently only support Unix-based systems, including Linux and macOS. 
For Windows machines, we suggest using the [Windows Subsystem for Linux (WSL)](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux).

The package requires `numpy`, `scipy`, `scikit-learn`, `cython` and `jax`, and `matplotlib` for the visualizations.

The package contains Cython-generated C extensions that are automatically compiled during installation. 

The latest release is available through pip:

```sh
pip install dadapy
```

To install the latest development version, clone the source code from GitHub
and install it with pip as follows:

```sh
pip install git+https://github.com/sissa-data-science/DADApy
```

Alternatively, if you'd like to modify the implementation of some function locally you can download the repository and install the package with:

```sh
git clone https://github.com/sissa-data-science/DADApy.git
cd DADApy
python setup.py build_ext --inplace
pip install .
```

The methods of the class ```DiffImbalance``` can be run on GPU, using a suitable installation of JAX on a GPU platform. The code has been tested with JAX v0.4.16 with CUDA 12, which can be installed with:

```sh
pip install --upgrade "jax[cuda12_pip]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For more information on the installation of the JAX library on GPUs see the official [repository](https://github.com/google/jax?tab=readme-ov-file#installation).

# Citing DADApy

A description of the package is available [here](https://www.sciencedirect.com/science/article/pii/S2666389922002070).

Please consider citing it if you found this package useful for your research:

```bib
@article{dadapy,
    title = {DADApy: Distance-based analysis of data-manifolds in Python},
    journal = {Patterns},
    pages = {100589},
    year = {2022},
    issn = {2666-3899},
    doi = {https://doi.org/10.1016/j.patter.2022.100589},
    url = {https://www.sciencedirect.com/science/article/pii/S2666389922002070},
    author = {Aldo Glielmo and Iuri Macocco and Diego Doimo and Matteo Carli and Claudio Zeni and Romina Wild and Maria d’Errico and Alex Rodriguez and Alessandro Laio},
    }
```
