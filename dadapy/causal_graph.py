# Copyright 2021-2024 The DADApy Authors. All Rights Reserved.
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

"""
The *causal_graph* module contains the *CausalGraph* class, which inherits from the *DiffImbalance* class.

The code can be runned on gpu using the command
    jax.config.update('jax_platform_name', 'gpu') # set 'cpu' or 'gpu'
"""

import numpy as np
from dadapy import DiffImbalance
from tqdm.auto import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import string
import warnings


class CausalGraph(DiffImbalance):
    """Constructs a coarse-grained causal graph where variables are grouped into single nodes.

    Attributes:
        time_series (np.ndarray(float)): array of shape (N_times,D), where N_times is the length
            of trajectory and D is the number of dynamical variables. The sampling time is supposed 
            to be constant along the trajectory and for all the variables.
        periods (np.ndarray(float)): array of shape (D,) containing the periods of the dynamical variables.
            The default is None, which means that the variables are treated as nonperiodic. If only some 
            variables are periodic, the entry of the nonperiodic ones should be set to 0.
        seed (int): seed of jax random generator
    """
    def __init__(
        self,
        time_series,
        periods=None,
        seed=0,
    ):
        self.time_series = time_series
        self.periods = (
            np.ones(time_series.shape[1]) * np.array(periods)
            if periods is not None
            else periods
        )
        self.seed = seed
        self.num_variables = self.time_series.shape[1]
        self.imbs = None
        self.weights = None
        self.adj_matrix = None
        self.groups_dictionary = None

    def return_nn_indices(
        self,
        variables,
        num_samples,
        time_lags,
        discard_close_ind,
    ):
        """
        Returns the indices of the nearest neighbors of each point, given the sampling method.

        Args:
            variables (list, jnp.array(int)): array of the coordinates used to build the distance space (with weights 1)
            num_samples (int): number of samples harvested from the full time series
            time_lags (list(int), np.array(int)): tested time lags between 'present' and 'future'
            discard_close_ind (int): defines the "close points" for which distances and ranks are not computed: for each point i, 
                the distances d[i,i-discard_close_ind:i+discard_close_ind+1] are discarded.
        Returns:
            nn_indices (np.array(float)): array of the nearest neighbors indices: nn_indices[i] is the index of the column
                with value 1 in the rank matrix
        """
        assert num_samples < self.time_series.shape[0]-max(time_lags), (
            f"Error: cannot extract {num_samples} samples from {self.time_series.shape[0]} initial samples, "
            +f"if the maximum time lag is {max(time_lags)}.\nChoose a value of num_samples such that "
            +f"num_samples < {self.time_series.shape[0]} - {max(time_lags)}"
        )
        indices_present = np.linspace(0, # select times defining the ensemble of trajectories
                                      self.time_series.shape[0]-max(time_lags)-1, 
                                      num_samples, dtype=int) 
        coords_present = self.time_series[indices_present]
        dii = DiffImbalance(
            data_A=coords_present,
            data_B=coords_present,
            discard_close_ind=discard_close_ind
        )
        nn_indices = dii._return_nn_indices(variables=variables)
        return nn_indices

    def optimize_present_to_future(
        self,
        num_samples,
        time_lags,
        target_variables="all",
        num_epochs=100,
        batches_per_epoch=1,
        l1_strength=0.0,
        point_adapt_lambda=False,
        k_init=1,
        k_final=1,
        lambda_init=None,
        lambda_final=None,
        init_params=None,
        optimizer_name="sgd",
        learning_rate=1e-1,
        learning_rate_decay=True,
        compute_error=False,
        ratio_rows_columns=1,
        num_points_rows=None,
        discard_close_ind=None
    ):
        """
        Iteratively optimizes the DII from the full space in the present to a target space in the future.
    
        Args:
            num_samples (int): number of samples harvested from the full time series, interpreted as
                independent initial conditions of the same dynamical process
            time_lags (list(int), np.ndarray(int)): tested time lags between 'present' and 'future'
            target_variables (str or list(int), np.array(int)): list or np.array of the target variables
                defining the distance space in the future. By default target_variables=="all", which means 
                that the optimization is iterated over all the variables as target.
            num_epochs (int): number of training epochs
            batches_per_epoch (int): number of minibatches; must be a divisor of n_points. Each update of the weights is
                carried out by computing the gradient over n_points / batches_per_epoch points. The default is 1, which
                means that the gradient is computed over all the available points (batch GD).
            l1_strength (float): strength of the L1 regularization (LASSO) term. The default is 0.
            point_adapt_lambda (bool): whether to use a global smoothing parameter lambda for the c_ij coefficients
                in the DII (if False), or a different parameter for each point (if True). The default is False.
            k_init (int): initial rank of the neighbors used to set lambda. The default is 1.
            k_final (int): initial rank of the neighbors used to set lambda. The default is 1.
            lambda_init (float): initial value of lambda
            lambda_final (float): final value of lambda
            init_params (np.array(float), jnp.array(float)): array of shape (n_features_A,) containing the initial
                values of the scaling weights to be optimized. If None, init_params == [0.1, 0.1, ..., 0.1].
            optimizer_name (str): name of the optimizer, calling the Optax library. The possible choices are 'sgd'
                (default), 'adam' and 'adamw'. See https://optax.readthedocs.io/en/latest/api/optimizers.html for
                more.
            learning_rate (float): value of the learning rate. The default is 1e-1.
            learning_rate_decay (str): schedule to damp the learning rate to zero starting from the value provided 
                with the attribute learning_rate. The avilable schedules are: cosine decay ("cos"), exponential
                decay ("exp"; the initial learning rate is halved every 10 steps), or constant learning rate (None).
                The default is "cos".
            compute_error (bool): whether to compute the standard Information Imbalance, if False (default), or to
                compute distances between points in two different groups and return the error associated to the DII
                during the training, if True
            ratio_rows_columns (float): only read when compute_error == True, defines the ratio between the number
                of points along the rows and the number points along the columns of distance and rank matrices, in two
                groups randomly sampled. The default is 1, which means that the two groups are constructed with
                n_points / 2 and n_points / 2 points.
            num_points_rows (int): number of points sampled from the rows of rank and distance matrices. In case of large
                datasets, choosing num_points_rows < n_points can significantly speed up the training. The default is
                None, for which num_points_rows == n_points.
            discard_close_ind (int): defines the "close points" (following the same labelling order of data_A and
                data_B, along axis=0) for which distances and ranks are not computed: for each point i, the distances 
                d[i,i-discard_close_ind:i+discard_close_ind+1] are discarded. This option is only available with
                batches_per_epoch=1, compute_error=False and num_points_rows=None. The default is None, for which no 
                "close points" are discarded.

        Returns:
            weights_final (np.array(float)): array of shape (n_target_variables,n_time_lags,D) containing the
                D final scaling weights for each optimization, where D is the number of variables in the time series
            imbs (np.array(float)): array of shape (n_target_variables, n_time_lags, num_epochs+1) containing the DII 
                during the whole trainings
        """
        assert num_samples < self.time_series.shape[0]-max(time_lags), (
            f"Error: cannot extract {num_samples} samples from {self.time_series.shape[0]} initial samples, "
            +f"if the maximum time lag is {np.max(time_lags)}.\nChoose a smaller value of num_samples."
        )
        indices_present = np.linspace(0, # select times defining the ensemble of trajectories
                                      self.time_series.shape[0]-max(time_lags)-1, 
                                      num_samples, dtype=int)        
        coords_present = self.time_series[indices_present]
        if target_variables == "all":
            target_variables = np.arange(self.num_variables)

        imbs = np.zeros((len(target_variables),len(time_lags),num_epochs+1))
        weights_final = np.zeros((len(target_variables),len(time_lags),self.num_variables)) # only final weights saved; may be modified
        # loop over target variables and time lags
        for i_var, target_var in enumerate(target_variables):
            for j_tau, tau in enumerate(time_lags):
                coords_future = self.time_series[indices_present+tau,target_var].reshape((-1,1))

                dii = DiffImbalance(
                    data_A=coords_present,
                    data_B=coords_future,
                    periods_A=self.periods, 
                    periods_B=None if self.periods is None else self.periods[target_var], 
                    seed=self.seed,
                    num_epochs=num_epochs,
                    batches_per_epoch=batches_per_epoch,
                    l1_strength=l1_strength,
                    point_adapt_lambda=point_adapt_lambda,
                    k_init=k_init,
                    k_final=k_final,
                    lambda_init=lambda_init,
                    lambda_final=lambda_final,
                    init_params=init_params,
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate,
                    learning_rate_decay=learning_rate_decay,
                    compute_error=compute_error,
                    ratio_rows_columns=ratio_rows_columns,
                    num_points_rows=num_points_rows,
                    discard_close_ind=discard_close_ind
                )
                if compute_error:
                    weights, imbs[i_var,j_tau], _ = dii.train(bar_label=f"target_var={target_var}, tau={tau}")
                else:
                    weights, imbs[i_var,j_tau] = dii.train(bar_label=f"target_var={target_var}, tau={tau}")
                weights_final[i_var,j_tau] = weights[-1] # save final weights only
        
        self.weights = weights_final
        self.imbs = imbs
        return weights_final, imbs
        
    def compute_adj_matrix(self, weights=None, threshold=1e-1):
        """
        Computes the adjacency matrix from the optimized weights produced by the method optimize_present_to_future.

        As a preliminary step before applying the threshold, the maximum weight over the tested time lags is taken
        for each pair X_i(0) -> X_j(tau) (i,j=1,...,D)

        Args:
            weights (np.ndarray(float)): array of shape (D,n_time_lags,D) containing the optimal scaling weights 
                produced by optimize_present_to_future with the option target_variables="all". The default is None,
                which means that the weights are initialized from the last optimization.
            threshold (float): value of the threshold used to construct the adjacency matrix. If a weight is smaller
                than the threshold the corresponding entry in the adjacency matrix is set to 0, otherwise it is set to 1.

        Returns:
            adj_matrix (np.ndarray(float)): array of shape (D,D) defining the adjacency matrix of a directed graph, where
                each arrow defines a direct or indirect link
        """
        if weights is None:
            assert self.weights is not None, (
                "Error: first perform the optimization using the method optimize_present_to_future, or "
                +"provide a valid input for the 'weights' argument",
            )
            weights = self.weights
            warnings.warn(
                f"The adjacency matrix will be constructed with the weights of the last optimization"
            )
        assert weights.shape[0] == self.num_variables, (
            "Error: first perform the optimization using all variables as target, setting "
            +"target_variables='all'"
        )
        weights_max = np.max(weights, axis=1) # select maximum weights over all tested time lags
        adj_matrix = np.zeros((self.num_variables, self.num_variables)) 
        adj_matrix[weights_max.T > threshold] = 1 # apply threshold
        self.adj_matrix = adj_matrix
        return adj_matrix
    
    def _ancestors(self, adj_matrix=None): # this replaces function "ancestors" coded by Matteo
        """
        Finds the ancestors of each node in the directed graph described by the input adjacency matrix.

        Args:
            adj_matrix (np.ndarray(float)): binary matrix of shape (D,D) defining the links of a directed 
                graph. If None, the matrix constructed from the last call of the method 'compute_adj_matrix'
                is employed.

        Returns:
            min_auto_sets (list): list of lists, such that min_auto_sets[i] contains the indices of all the 
                ancestors of node i in the graph
        """
        if adj_matrix is None:
            assert self.adj_matrix is not None, (
                "Error: first compute the adjacency matrix with the method compute_adj_matrix"
            )
            adj_matrix = self.adj_matrix
        G = nx.DiGraph(adj_matrix)
        min_auto_sets = []
        for var in np.arange(adj_matrix.shape[0]):
            min_auto_sets.append( sorted(nx.ancestors(G, var) | {var}) )
        return min_auto_sets

    def find_groups(self, adj_matrix=None):
        """
        Finds the groups of variables defining new nodes in the coarse-grained causal graph.

        Args:
            adj_matrix (np.ndarray(float)): binary matrix of shape (D,D) defining the links of a directed 
                graph with D nodes. If None, the matrix constructed from the last call of the method 
                'compute_adj_matrix' is employed.

        Returns:
            groups_dictionary (dict): dictionary with pairs (group_id, order) as keys and lists containing
                the indices of the variables in each group as values. group_id is an integer number identifying
                the group, while order is an integer identifying the step of the algorithm at which the group is
                identified, namely its level of autonomy. Both group_id and order start from 0.
        """
        if adj_matrix is None:
            assert self.adj_matrix is not None, (
                "Error: first compute the adjacency matrix with the method compute_adj_matrix"
            )
            adj_matrix = self.adj_matrix
        
        min_auto_sets = self._ancestors(adj_matrix)
        group_index = 0
        order_index = 0
        variables_assigned = []
        groups_dictionary = {}
        min_auto_sets_left = min_auto_sets.copy()

        while len(min_auto_sets_left) != 0:
            nvariables_left = len(min_auto_sets_left)
            sets_sizes = [len(min_auto_sets_left[i_group]) for i_group in range(nvariables_left)]
            smallest_set_index = np.argmin(sets_sizes)
            groups_dictionary[group_index,order_index] = min_auto_sets_left[smallest_set_index]
            variables_assigned.extend(min_auto_sets_left[smallest_set_index])
            min_auto_sets_left.pop(smallest_set_index) # delete group from list of minimal autonomous subsets
            group_index += 1

            parallel_groups = 0
            min_auto_sets_left_temp = min_auto_sets_left.copy()
            for try_set_index, try_set in enumerate(min_auto_sets_left):
                intersection = set(variables_assigned).intersection(try_set)
                if intersection == set():
                    groups_dictionary[group_index,order_index] = min_auto_sets_left[try_set_index]
                    variables_assigned.extend(min_auto_sets_left[try_set_index])
                    min_auto_sets_left_temp.pop(try_set_index-parallel_groups) # delete group from list of minimal autonomous subsets
                    group_index += 1
                    parallel_groups += 1
            min_auto_sets_left = min_auto_sets_left_temp.copy()

            for left_set_index, left_set in enumerate(min_auto_sets_left):
                min_auto_sets_left[left_set_index] = list(
                    set(left_set).difference(variables_assigned)
                )
            # delete empty lists
            min_auto_sets_left = [set_left for set_left in min_auto_sets_left if len(set_left) > 0]
            order_index += 1
        
        self.groups_dictionary = groups_dictionary
        return groups_dictionary


    def community_graph_visualization(self, groups_dictionary=None, adj_matrix=None): #features, metafeatures, adjacency):
        '''Plots a visual representation of the coarse-grained causal graph

        Args:
            groups_dictionary (dict): dictionary with pairs (group_id,level) as keys and lists containing
                the indices of the variables in each group as values. If None, the output of the last call
                of 'find_groups' is employed.
            adj_matrix (np.array(float)): matrix of shape (D,D) defining the links between the variables 
                after thresholding the matrix of the optimized weights. If None, the output of the last call
                of 'compute_adj_matrix' is employed.
        '''
        # construct graph
        G = nx.DiGraph()
        keys = list(groups_dictionary.keys())
        values = list(groups_dictionary.values())
        alphabet_string = list(string.ascii_uppercase)
        group_names = {tuple(group): alphabet_string[i] for i, group in enumerate(values)}
        from_names_to_groups = {group_names[key]: key for key in group_names.keys()}

        # convert groups into names and add them to graph as nodes
        for group, key in zip(group_names, keys):
            group_name = group_names[tuple(group)]
            G.add_node(str(group_name))
            #sample_var_from_group = from_names_to_groups[group_name][0]
            print(f"Group {group_name} ({len(group)} variables, order {key[1]}): {group}")

        # dictionary with keys: (order_idx) and values: list of groups at that order (list of list)
        groups_orders = {key[1]: [] for key in keys} #{order_idx: [] for order_idx in orders_idx}
        for group_idx, order_idx in keys:
            groups_orders[order_idx].append(groups_dictionary[group_idx, order_idx])
        # draw edges
        for group_effect_idx, order_idx in keys:
            if order_idx > 0:
                # for each putative effect group at order >=1...
                for group_effect in groups_orders[order_idx]:
                    group_name_effect = group_names[tuple(group_effect)]
                    # ...loop over all putative cause groups at order -1, -2, ...
                    for previous_order in range(1,order_idx+1):
                        for group_cause in groups_orders[order_idx-previous_order]:
                            group_name_cause = group_names[tuple(group_cause)]
                            effect_ancestors_names = set(nx.ancestors(G,group_name_effect))
                            effect_ancestors_set = [
                                from_names_to_groups[effect_ancestor_name] for effect_ancestor_name in effect_ancestors_names
                            ]
                            effect_ancestors_set = set().union(*effect_ancestors_set)
                            group_cause_not_already_ancestor = set(group_cause).difference(effect_ancestors_set)
                            # ...loop over all variables in each putative cause group, avoiding groups that are already ancestors...
                            for variable_cause in group_cause_not_already_ancestor:
                                # ...and draw an edge if at least a link is found
                                if adj_matrix[variable_cause,group_effect].any():
                                    G.add_edges_from([ (str(group_name_cause), str(group_name_effect)) ])
                                    break
        # show graph
        options = {
            'node_color': 'gray',
            'node_size': 3000,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        nx.draw_circular(G, arrows=True, with_labels=True, **options)
        plt.show()