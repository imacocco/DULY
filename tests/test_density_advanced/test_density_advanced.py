# Copyright 2021-2023 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for testing the DensityAdvanced class."""

import os

import numpy as np

from dadapy import DensityAdvanced

# define a basic dataset with 6 points
data = np.array([[0, 0], [0.15, 0], [0.2, 0], [4, 0], [4.1, 0], [4.2, 0]])


def test_compute_grads():
    """Test the compute_grads method."""
    # define the expected gradient
    expected_grads = np.array(
        [
            [13.124999999999995, 0],
            [-6.66666666666666, 0.0],
            [-9.374999999999998, 0.0],
            [11.249999999999973, 0.0],
            [1.3322676295501972e-13, 0.0],
            [-11.250000000000007, 0.0],
        ]
    )

    expected_grad_vars = [
        [2.6367187500000426, 0.0],
        [133.33333333333337, 0.0],
        [23.73046874999999, 0.0],
        [10.546875000000107, 0.0],
        [675.0000000000107, 0.0],
        [10.546874999999883, 0.0],
    ]

    da = DensityAdvanced(coordinates=data, maxk=3, verbose=True)
    da.compute_distances()
    da.set_id(1)
    da.set_kstar(3)
    da.compute_grads(comp_covmat=False)

    assert np.allclose(da.grads, expected_grads)
    assert np.allclose(da.grads_var, expected_grad_vars)


expected_pearson_array = np.array([1.0 / 3.0, -1.0, -1.0, -1.0, -1.0, 1.0 / 3.0])


def test_compute_pearson():
    """Test the compute_pearson method."""
    # create the DensityAdvanced object
    da = DensityAdvanced(coordinates=data, maxk=2, verbose=True)
    da.compute_distances()
    da.set_id(1)
    da.set_kstar(2)
    da.compute_pearson()
    assert np.allclose(da.pearson_array, expected_pearson_array)


def test_compute_deltaFs():
    """Test the compute_deltaFs method."""
    # define the expected deltaFs
    expected_Fij_array = np.array(
        [5.999999999999997, 0.0, 0.0, 0.0, 0.0, 3.0000000000000133]
    )
    expected_Fij_var_array = np.array(
        [1.5017050605028698e-12, 0.0, 0.0, 0.0, 0.0, 6.666666666666738e-13]
    )

    da = DensityAdvanced(coordinates=data, maxk=2, verbose=True)
    da.compute_distances()
    da.set_id(1)
    da.set_kstar(2)
    da.compute_deltaFs()
    assert np.allclose(da.Fij_array, expected_Fij_array)
    assert np.allclose(da.Fij_var_array, expected_Fij_var_array)


expected_density_BMTI = np.array(
    [
        -1.744691095848123652,
        -3.116998743804390681,
        -2.572142495016555674,
        -3.486578399734634459,
        -1.846995010314064212,
        -3.048453200003461205,
        -3.960306993019918842,
        -1.837393076792057212,
        -1.724734740892750251,
        -3.924494011546748151,
        -2.197128574850228944,
        -3.131326434273491888,
        -3.685895881530738993,
        -4.178576239180523899,
        -1.704307784613914079,
        -2.483044453006292951,
        -1.990341566326547351,
        -3.028532376683757743,
        -2.929863713536639214,
        -2.837116806618219744,
        -3.653357026225191095,
        -3.107871433163512886,
        -2.215995465474911885,
        -2.636287799689599254,
        -2.577608131196509778,
    ]
)


def test_density_BMTI():
    """Test the density_BMTI method."""
    filename = os.path.join(os.path.split(__file__)[0], "../2gaussians_in_2d.npy")

    X = np.load(filename)[:25]

    da = DensityAdvanced(coordinates=X, maxk=10, verbose=True)
    da.compute_distances()
    da.set_id(2)
    da.compute_density_BMTI(alpha=0.99)

    assert np.allclose(da.log_den, expected_density_BMTI, rtol=1e-05, atol=1e-01)
