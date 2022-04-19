# Copyright 2021-2022 The DADApy Authors. All Rights Reserved.
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

"""This module implements contains the implementation of the IdEstimation class."""

import copy
import math
import multiprocessing
from functools import partial

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import pairwise_distances_chunked

from dadapy._base import Base
from dadapy.utils_ import utils as ut
from dadapy.utils_.utils import compute_nn_distances

cores = multiprocessing.cpu_count()
rng = np.random.default_rng()


class IdEstimation(Base):
    """IdEstimation class."""

    def __init__(
        self,
        coordinates=None,
        distances=None,
        maxk=None,
        period=None,
        verbose=False,
        njobs=cores,
    ):
        """Estimate the intrinsic dimension of a dataset choosing among various routines.

        Inherits from class Base.

        Args:
            coordinates (np.ndarray(float)): the data points loaded, of shape (N , dimension of embedding space)
            distances (np.ndarray(float)): A matrix of dimension N x mask containing distances between points
            maxk (int): maximum number of neighbours to be considered for the calculation of distances
            period (np.array(float), optional): array containing periodicity for each coordinate. Default is None
            verbose (bool): whether you want the code to speak or shut up
            njobs (int): number of cores to be used
        """
        super().__init__(
            coordinates=coordinates,
            distances=distances,
            maxk=maxk,
            period=period,
            verbose=verbose,
            njobs=njobs,
        )

        self.intrinsic_dim = None
        self.intrinsic_dim_err = None
        self.intrinsic_dim_scale = None

    # ----------------------------------------------------------------------------------------------

    def _compute_id_2NN(self, mus, fraction, algorithm="base"):
        """Compute the id using the 2NN algorithm.

        Helper of return return_id_2NN.

        Args:
            mus (np.ndarray(float)): ratio of the distances of first- and second-nearest neighbours
            fraction (float): fraction of mus to take into account, discard the highest values
            algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood

        Returns:
            intrinsic_dim (float): the estimation of the intrinsic dimension

        """
        N = mus.shape[0]
        N_eff = int(N * fraction)
        mus_reduced = np.sort(mus)[:N_eff]

        # TO FIX: maximum likelihood should be computed with the unbiased estimator N-1/log(mus)?
        if algorithm == "ml":
            intrinsic_dim = N / np.sum(mus)

        elif algorithm == "base":
            y = -np.log(1 - np.arange(1, N_eff + 1) / N)

            def func(x, m):
                return m * x

            intrinsic_dim, _ = curve_fit(func, mus_reduced, y)

        else:
            raise ValueError("Please select a valid algorithm type")

        return intrinsic_dim

    # ----------------------------------------------------------------------------------------------
    def compute_id_2NN(
        self, algorithm="base", fraction=0.9, decimation=1, set_attr=True
    ):
        """Compute intrinsic dimension using the 2NN algorithm.

        Args:
            algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood
            fraction (float): fraction of mus that will be considered for the estimate (discard highest mus)
            decimation (float): fraction of randomly sampled points used to compute the id
            set_attr (bool): whether to change the class attributes as a result of the computation

        Returns:
            id (float): the estimated intrinsic dimension
            id_err (float): the standard error on the id estimation
            rs (float): the average nearest neighbor distance (rs)

        References:
            E. Facco, M. d’Errico, A. Rodriguez, A. Laio, Estimating the intrinsic dimension of datasets by a minimal
            neighborhood information, Scientific reports 7 (1) (2017) 1–8

        """
        nrep = int(np.rint(1.0 / decimation))
        ids = np.zeros(nrep)
        rs = np.zeros(nrep)

        for j in range(nrep):

            if decimation == 1 and self.distances is not None:
                # with decimation == 1 use saved distances if present
                distances, dist_indices = self.distances, self.dist_indices

            elif decimation == 1 and self.distances is None and set_attr is True:
                # with decimation ==1 and set_attr==True compute distances and save them
                self.compute_distances()
                distances, dist_indices = self.distances, self.dist_indices

            else:
                # if set_attr==False or for decimation < 1 random sample points don't save distances
                N_subset = int(np.rint(self.N * decimation))
                idx = np.random.choice(self.N, size=N_subset, replace=False)
                X_decimated = self.X[idx]

                distances, dist_indices = compute_nn_distances(
                    X_decimated,
                    maxk=3,  # only compute first 2 nn
                    metric=self.metric,
                    period=self.period,
                )

            mus = np.log(distances[:, 2] / distances[:, 1])
            ids[j] = self._compute_id_2NN(mus, fraction, algorithm)
            rs[j] = np.mean(distances[:, np.array([1, 2])])

        intrinsic_dim = np.mean(ids)
        intrinsic_dim_err = np.std(ids) / len(ids) ** 0.5
        intrinsic_dim_scale = np.mean(rs)

        if self.verb:
            print(f"ID estimation finished: selecting ID of {intrinsic_dim}")

        if set_attr:
            self.intrinsic_dim = intrinsic_dim
            self.intrinsic_dim_err = intrinsic_dim_err
            self.intrinsic_dim_scale = intrinsic_dim_scale

        return intrinsic_dim, intrinsic_dim_err, intrinsic_dim_scale

    # ----------------------------------------------------------------------------------------------

    def return_id_scaling_2NN(
        self,
        N_min=10,
        algorithm="base",
        fraction=0.9,
    ):
        """Compute the id at different scales using the 2NN algorithm.

        Args:
            N_min (int): minimum number of points considered when decimating the dataset,
                        N_min effectively sets the largest 'scale';
            algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood;
            fraction (float): fraction of mus that will be considered for the estimate (discard highest mus).

        Returns:
            ids_scaling (np.ndarray(float)): array of intrinsic dimensions;
            ids_scaling_err (np.ndarray(float)): array of error estimates;
            rs_scaling (np.ndarray(float)): array of average distances of the neighbors involved in the estimates.

        Quick Start:
        ===========

        .. code-block:: python

                from dadapy import IdEstimation
                from sklearn.datasets import make_swiss_roll

                #two dimensional curved manifold embedded in 3d with noise

                n_samples = 5000
                X, _ = make_swiss_roll(n_samples, noise=0.3)

                ie = IdEstimation(coordinates=X)
                ids_scaling, ids_scaling_err, rs_scaling = ie.return_id_scaling_2NN(N_min = 20)

                ids_scaling:
                array([2.88 2.77 2.65 2.42 2.22 2.2  2.1  2.23])

                ids_scaling_err:
                array([0.   0.02 0.05 0.04 0.04 0.03 0.04 0.04])

                rs_scaling:
                array([0.52 0.66 0.88 1.18 1.65 2.3  3.23 4.54])
        """

        max_ndec = int(math.log(self.N, 2)) - 1
        Nsubsets = np.round(self.N / np.array([2**i for i in range(max_ndec)]))
        Nsubsets = Nsubsets.astype(int)

        if N_min is not None:
            Nsubsets = Nsubsets[Nsubsets > N_min]

        ids_scaling = np.zeros(Nsubsets.shape[0])
        ids_scaling_err = np.zeros(Nsubsets.shape[0])
        rs_scaling = np.zeros((Nsubsets.shape[0]))

        for i, N_subset in enumerate(Nsubsets):

            ids_scaling[i], ids_scaling_err[i], rs_scaling[i] = self.compute_id_2NN(
                algorithm=algorithm,
                fraction=fraction,
                decimation=N_subset / self.N,
                set_attr=False,
            )

        return ids_scaling, ids_scaling_err, rs_scaling

    # ----------------------------------------------------------------------------------------------
    def return_id_scaling_gride(self, range_max=64, d0=0.001, d1=1000, eps=1e-7):
        """Compute the id at different scales using the Gride algorithm.

        Args:
            range_max (int): maximum nearest neighbor rank considered for the id computations;
                            the number of id estimates are log2(range_max) as the nearest neighbor
                            order ('scale') is doubled at each estimate;
            d0 (float): minimum intrinsic dimension considered in the search;
            d1 (float): maximum intrinsic dimension considered in the search;
            eps (float): precision of the approximate id calculation.

        Returns:
            ids_scaling (np.ndarray(float)): array of intrinsic dimensions of length log2(range_max);
            ids_scaling_err (np.ndarray(float)): array of error estimates;
            rs_scaling (np.ndarray(float)): array of average distances of the neighbors involved in the estimates.

        Quick Start:
        ===========

        .. code-block:: python

                from dadapy import IdEstimation
                from sklearn.datasets import make_swiss_roll

                #two dimensional curved manifold embedded in 3d with noise

                n_samples = 5000
                X, _ = make_swiss_roll(n_samples, noise=0.3)

                ie = IdEstimation(coordinates=X)
                ids_scaling, ids_scaling_err, rs_scaling = ie.return_id_scaling_gride(range_max = 512)

                ids_scaling:
                array([2.81 2.71 2.48 2.27 2.11 1.98 1.95 2.05])

                ids_scaling_err:
                array([0.04 0.03 0.02 0.01 0.01 0.01 0.   0.  ])

                rs_scaling:
                array([0.52 0.69 0.93 1.26 1.75 2.48 3.54 4.99])


        References:
            F. Denti, D. Doimo, A. Laio, A. Mira, Distributional results for model-based intrinsic dimension
            estimators, arXiv preprint arXiv:2104.13832 (2021).
        """
        max_rank = min(self.N, range_max)
        max_step = int(math.log(max_rank, 2))
        nn_ranks = np.array([2**i for i in range(max_step)])

        if self.distances is not None and range_max < self.maxk + 1:
            max_rank = min(max_rank, self.maxk + 1)
            if self.verb:
                print(
                    f"distance already computed up to {max_rank}. max rank set to {max_rank}"
                )

            mus = self.distances[:, nn_ranks[1:]] / self.distances[:, nn_ranks[:-1]]
            rs = self.distances[:, np.array([nn_ranks[:-1], nn_ranks[1:]])]

        elif self.X is not None:

            if self.verb:
                print(
                    f"distance not computed up to {max_rank}. distance computation started"
                )

            distances, dist_indices, mus, rs = self._return_mus_scaling(
                range_scaling=max_rank
            )
            # returns:
            # distances, dist_indices of shape (self.N, self.maxk+1): sorted distances and dist indices up to maxk+1
            # mus of shape (self.N, len(nn_ranks)): ratio between 2*kth and kth neighbor distances of every data point
            # rs of shape (self.N, 2, len(nn_ranks)): kth, 2*kth neighbor of every data for kth in nn_ranks
            if self.verb:
                print("distance computation finished")

            # if distances have not been computed save them
            if self.distances is None:
                self.distances = distances
                self.dist_indices = dist_indices
                self.N = distances.shape[0]

        # array of ids (as a function of the average distance to a point)
        ids_scaling = np.zeros(mus.shape[1])
        # array of error estimates (via fisher information)
        ids_scaling_err = np.zeros(mus.shape[1])
        "average of the kth and 2*kth neighbor distances taken over all datapoints for each id estimate"
        rs_scaling = np.mean(rs, axis=(0, 1))

        # compute IDs (and their error) via maximum likelihood for all the scales up to max_rank
        if self.verb:
            print("id inference started")
        for i in range(mus.shape[1]):
            n1 = 2**i
            id = ut._argmax_loglik(
                self.dtype, d0, d1, mus[:, i], n1, 2 * n1, self.N, eps=eps
            )  # eps=precision id calculation
            ids_scaling[i] = id

            ids_scaling_err[i] = (
                1
                / ut._fisher_info_scaling(
                    id, mus[:, i], n1, 2 * n1, eps=5 * self.eps
                )  # eps=regularization small numbers
            ) ** 0.5
        if self.verb:
            print("id inference finished")

        return ids_scaling, ids_scaling_err, rs_scaling

    # ----------------------------------------------------------------------------------------------
    def _mus_scaling_reduce_func(self, dist, range_scaling):
        """Help to compute the "mus" needed to compute the id.

        Applied at the end of pairwise_distance_chunked see:
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/pairwise.py#L1474

        Once a chunk of the distance matrix is computed _mus_scaling_reduce_func
        1) estracts the distances of the  neighbors of order 2**i up to the maximum
        neighbor range given by range_scaling
        2) computes the mus[i] (ratios of the neighbor distance of order 2**(i+1)
        and 2**i (see return id scaling gride)
        3) returns the chunked distances up to maxk, the mus, and rs, the distances
        of the neighbors involved in the estimate

        Args:
            dist: chunk of distance matrix passed internally by pairwise_distance_chunked
            range_scaling (int): maximum neighbor rank

        Returns:
            dist: CHUNK of distance matrix sorted in increasing order of neighbor distances up to maxk
            neighb_ind: indices of the nearest neighbors up to maxk
            mus: ratios of the neighbor distances of order 2**(i+1) and 2**i
            rs: distances of the neighbors involved in the mu estimates
        """
        # argsort may be faster than argpartition when gride is applied on the full dataset (for the moment not used)

        max_step = int(math.log(range_scaling, 2))
        steps = np.array([2**i for i in range(max_step)])

        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, range_scaling - 1, axis=1)
        neigh_ind = neigh_ind[:, :range_scaling]

        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]

        dist = np.sqrt(dist[sample_range, neigh_ind])
        dist = self._remove_zero_dists(dist)
        mus = dist[:, steps[1:]] / dist[:, steps[:-1]]
        rs = dist[:, np.array([steps[:-1], steps[1:]])]

        dist = copy.deepcopy(dist[:, : self.maxk + 1])
        neigh_ind = copy.deepcopy(neigh_ind[:, : self.maxk + 1])

        return dist, neigh_ind, mus, rs

    def _return_mus_scaling(self, range_scaling):
        """Return the "mus" needed to compute the id.

        Adapted from kneighbors function of sklearn
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/neighbors/_base.py#L596
        It allows to keep a nearest neighbor matrix up to rank 'maxk' (few tens of points)
        instead of 'range_scaling' (few thousands), while computing the ratios between neighbors' distances
        up to neighbors' rank 'range scaling'.
        For big datasets it avoids out of memory errors

        Args:
            range_scaling (int): maximum neighbor rank considered in the computation of the mu ratios

        Returns:
            dist (np.ndarray(float)): the FULL distance matrix sorted in increasing order of distances up to maxk
            neighb_ind np.ndarray(int)): the FULL matrix of the indices of the nearest neighbors up to maxk
            mus np.ndarray(float)): the FULL matrix of the ratios of the neighbor distances of order 2**(i+1) and 2**i
            rs np.ndarray(float)): the FULL matrix of the distances of the neighbors involved in the mu estimates
        """
        reduce_func = partial(
            self._mus_scaling_reduce_func, range_scaling=range_scaling
        )

        kwds = {"squared": True}
        chunked_results = list(
            pairwise_distances_chunked(
                self.X,
                self.X,
                reduce_func=reduce_func,
                metric=self.metric,
                n_jobs=self.njobs,
                working_memory=1024,
                **kwds,
            )
        )

        neigh_dist, neigh_ind, mus, rs = zip(*chunked_results)

        return (
            np.vstack(neigh_dist),
            np.vstack(neigh_ind),
            np.vstack(mus),
            np.vstack(rs),
        )

    # ----------------------------------------------------------------------------------------------

    def compute_id_2NN_wprior(self, alpha=2, beta=5, posterior_mean=True):
        """Compute the intrinsic dimension using a bayesian formulation of 2nn.

        Args:
            alpha (float): parameter of the Gamma prior
            beta (float): parameter of the Gamma prior
            posterior_mean (bool): whether to use the posterior mean as estimator,
                if False the posterior mode will be used

        Returns:
            id (float): the estimated intrinsic dimension
            id_err (float): the standard error on the id estimation
            rs (float): the average nearest neighbor distance (rs)
        """
        if self.distances is None:
            self.compute_distances()

        if self.verb:
            print(
                "ID estimation started, using alpha = {} and beta = {}".format(
                    alpha, alpha
                )
            )

        distances_used = self.distances
        sum_log_mus = np.sum(np.log(distances_used[:, 2] / distances_used[:, 1]))

        alpha_post = alpha + self.N
        beta_post = beta + sum_log_mus

        mean_post = alpha_post / beta_post
        std_post = np.sqrt(alpha_post / beta_post**2)
        mode_post = (alpha_post - 1) / beta_post

        if posterior_mean:
            self.intrinsic_dim = mean_post
        else:
            self.intrinsic_dim = mode_post

        self.intrinsic_dim_err = std_post
        self.intrinsic_dim_scale = np.mean(distances_used[:, np.array([1, 2])])

        return self.intrinsic_dim, self.intrinsic_dim_err, self.intrinsic_dim_scale

    # ----------------------------------------------------------------------------------------------

    def fix_rk(self, rk, ratio=None):
        """Compute the k points within the given rk and n points within given rn.

        For each point, computes the number self.k of points within a sphere of radius rk
        and the number self.n within an inner sphere of radius rn=rk*ratio. It also provides
        a mask to take into account those points for which the statistics might be wrong, i.e.
        k == self.maxk, meaning that all available points were selected. If self.maxk is equal
        to the number of points of the dataset no mask will be applied

        Args:
            rk (float): external shell radius
            ratio (float,optional): ratio between internal and external shell radii of the shells

        """
        # checks-in and initialisations
        if self.distances is None:
            self.compute_distances()

        self.set_rk(rk)

        if ratio is not None:
            self.set_r(ratio)
        else:
            assert (
                self.r is not None
            ), "set self.r or insert a value for the ratio parameter"

        # routine
        self.rn = self.rk * self.r
        self.k = (self.distances <= self.rk).sum(axis=1)
        self.n = (self.distances <= self.rn).sum(axis=1)

        # checks-out
        if self.maxk == self.N - 1:
            self.mask = np.ones(self.N, dtype=bool)
        else:
            # if not all available NN were taken into account (i.e. maxk < N) and k is equal to self.maxk
            # or distances[:,-1]<lk, it is likely that there are other points within lk that are not being
            # considered and thus potentially altering the statistics -> neglect them through self.mask
            # in the calculation of likelihood
            self.mask = self.distances[:, -1] > self.rk  # or self.k == self.maxk

            if np.any(~self.mask):
                print(
                    "NB: for "
                    + str(sum(~(self.mask)))
                    + " points, the counting of k could be wrong, "
                    + "as more points might be present within the selected Rk. In order not to affect "
                    + "the statistics a mask is provided to remove them from the calculation of the "
                    + "likelihood or posterior.\nConsider recomputing NN with higher maxk or lowering Rk."
                )

    # ----------------------------------------------------------------------------------------------

    def compute_id_binomial_rk(
        self, rk, ratio=None, subset=None, method="bayes", plot=False
    ):
        """Calculate the id using the binomial estimator by fixing the same eternal radius for all the points.

        In the estimation of the id one has to remove the central point from the counting of n and k
        as it is not effectively part of the poisson process generating its neighbourhood.

        Args:
            rk (float): radius of the external shell
            ratio (float): ratio between internal and external shell
            subset (int, np.ndarray(int)): choose a random subset of the dataset to make the id estimate
            method (str, default='bayes'): choose method between 'bayes' and 'mle'. The bayesian estimate
                gives the mean value and std of d, while mle only the max of the likelihood
            plot (bool, default=False): if True plots the posterior
                and initialise self.posterior_domain and self.posterior

        Returns:
            None

        """
        # checks-in and initialisations
        if ratio is not None:
            self.set_r(ratio)
        else:
            assert (
                self.r is not None
            ), "set self.r or insert a value for the ratio parameter"

        self.set_rk(rk)
        self.rn = self.rk * self.r

        # routine
        self.fix_rk(rk)

        n_eff = self.n[self.mask]
        k_eff = self.k[self.mask]

        # eventually perform the estimate using only a subset of the points (k and n are still computed on the whole
        # dataset!!!)
        if subset is not None:
            # if subset is integer draw that amount of random numbers
            if isinstance(subset, (np.integer, int)):
                if subset < self.mask.sum():
                    subset = rng.choice(
                        len(n_eff), subset, replace=False, shuffle=False
                    )
            # if subset is an array, use it as a mask
            elif isinstance(subset, np.ndarray):
                assert subset.shape[0] < self.mask.sum()
                assert isinstance(subset[0], (int, np.integer))
            else:
                print("choose a proper shape for the subset")
                return 0

            n_eff = n_eff[subset]
            k_eff = k_eff[subset]

        e_n = n_eff.mean()
        e_k = k_eff.mean()
        if e_n == 1.0:
            print(
                "no points in the inner shell, returning 0. Consider increasing rk and/or the ratio"
            )
            self.id_estimated_binom = 0
            return 0

        if method == "mle":
            self.id_estimated_binom = np.log((e_n - 1.0) / (e_k - 1.0)) / np.log(self.r)
            self.id_estimated_binom_std = (
                ut._compute_binomial_cramerrao(
                    self.id_estimated_binom, e_k - 1.0, self.r, n_eff.shape[0]
                )
                ** 0.5
            )

        elif method == "bayes":
            (
                self.id_estimated_binom,
                self.id_estimated_binom_std,
                self.posterior_domain,
                self.posterior,
            ) = ut._beta_prior(k_eff - 1, n_eff - 1, self.r, plot=plot)
        else:
            print("select a proper method for id computation")
            return 0

    # ----------------------------------------------------------------------------------------------

    def fix_k(self, k_eff=None, ratio=None):
        """Compute rk, rn, n for each point of the dataset given a value of k.

        This routine computes the external radius rk, internal radius rn and internal points n
        given a value k, the number of NN to consider.

        Args:
            k_eff (int, default=self.maxk): the number of NN to take into account
            ratio (float): ratio among rn and rk

        """
        # checks-in and initialisations
        if self.distances is None:
            self.compute_distances()

        if ratio is not None:
            self.set_r(ratio)
        else:
            assert (
                self.r is not None
            ), "set self.r or insert a value for the parameter ratio"

        if k_eff is not None:
            assert (
                k_eff < self.maxk
            ), "You first need to recompute the distances with the proper amount on NN"
        else:
            k_eff = self.maxk - 1

        # routine
        self.k = k_eff
        self.rk = self.distances[:, self.k]
        self.rn = self.rk * self.r
        self.n = (self.distances <= self.rn.reshape(self.N, 1)).sum(axis=1)
        # mask computed for consistency as in the fix_rk method, but no points should be ever excluded
        self.mask = np.ones(self.N, dtype=bool)

    # --------------------------------------------------------------------------------------

    def compute_id_binomial_k(
        self, k=None, ratio=None, subset=None, method="bayes", plot=False
    ):
        """Calculate id using the binomial estimator by fixing the number of neighbours.

        As in the case in which one fixes rk, also in this version of the estimation
        one removes the central point from n and k. Furthermore, one has to remove also
        the k-th NN, as it plays the role of the distance at which rk is taken.
        So if k=5 it means the 5th NN from the central point will be considered,
        taking into account 6 points though (the central one too). This means that
        in principle k_eff = 6, to which I'm supposed to subtract 2. For this reason
        in the computation of the MLE we have directly k-1, which explicitly would be k_eff-2

        Args:
            k (int): order of neighbour that set the external shell
            ratio (float): ratio between internal and external shell
            subset (int): choose a random subset of the dataset to make the Id estimate
            method (str, default='bayes'): choose method between 'bayes' and 'mle'. The bayesian estimate
                gives the mean value and std of d, while mle only the max of the likelihood
            plot (bool, default=False): if True plots the posterior
                and initialise self.posterior_domain and self.posterior

        Returns:
            None

        """
        # checks-in and initialisations
        if ratio is not None:
            self.set_r(ratio)
        else:
            assert (
                self.r is not None
            ), "set self.r or insert a value for the ratio parameter"

        if k is not None:
            assert (
                k < self.maxk
            ), "You first need to recompute the distances with the proper number of NN"
        else:
            k = self.maxk - 1

        # routine
        self.fix_k(k)
        n_eff = self.n

        # eventually perform the estimate using only a subset of the points (k and n are still computed on the whole
        # dataset!!!)
        if subset is not None:
            # if subset is integer draw that amount of random numbers
            if isinstance(subset, (np.integer, int)):
                if subset < self.N:
                    subset = rng.choice(
                        len(n_eff), subset, replace=False, shuffle=False
                    )
            # if subset is an array, use it as a mask
            elif isinstance(subset, np.ndarray):
                assert subset.shape[0] < self.N
                assert isinstance(subset[0], (int, np.integer))
            else:
                print("choose a proper shape for the subset")
                return 0

            n_eff = n_eff[subset]

        e_n = n_eff.mean()
        if e_n == 1.0:
            print(
                "no points in the inner shell, returning 0\n. Consider increasing rk and/or the ratio"
            )
            self.id_estimated_binom = 0
            return 0

        if method == "mle":
            self.id_estimated_binom = np.log((e_n - 1) / (k - 1)) / np.log(self.r)
            self.id_estimated_binom_std = (
                ut._compute_binomial_cramerrao(
                    self.id_estimated_binom, k - 1, self.r, n_eff.shape[0]
                )
                ** 0.5
            )
        elif method == "bayes":
            (
                self.id_estimated_binom,
                self.id_estimated_binom_std,
                self.posterior_domain,
                self.posterior,
            ) = ut._beta_prior(k - 1, n_eff - 1, self.r, plot=plot)
        else:
            print("select a proper method for id computation")
            return 0

    # ----------------------------------------------------------------------------------------------
    def set_id(self, d):
        """Set the intrinsic dimension."""
        assert d > 0, "cannot support negative dimensions (yet)"
        self.intrinsic_dim = d

    # ----------------------------------------------------------------------------------------------
    def set_r(self, r):
        """Set the r parameter."""
        assert 0 < r < 1, "select a proper ratio, 0<r<1"
        self.r = r

    # ----------------------------------------------------------------------------------------------
    def set_rk(self, R):
        """Set the rk parameter."""
        assert 0 < R, "select a proper rk>0"
        self.rk = R
