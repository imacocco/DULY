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
        time_series (np.array(float)): array of shape (N_times,D), where N_times is the length
            of trajectory and D is the number of dynamical variables. The sampling time is supposed 
            to be constant along the trajectory and for all the variables.
        coords_present (np.array(float)): array of shape (N_samples,D) containing the samples of the 
            D-dimensional trajectory at time t=0; read only when time_series==None
        coords_future (np.array(float)): array of shape (N_samples,D,n_lags) containing the samples of the 
            D-dimensional trajectory at different time lags; read only when time_series==None. If you want
            to test a single time lag, reshape the dataset with coords_future[:,:,np.newaxis]. 
        periods (np.ndarray(float)): array of shape (D,) containing the periods of the dynamical variables.
            The default is None, which means that the variables are treated as nonperiodic. If only some 
            variables are periodic, the entry of the nonperiodic ones should be set to 0.
        seed (int): seed of jax random generator
    """
    def __init__(
        self,
        time_series=None,
        coords_present=None,
        coords_future=None,
        periods=None,
        seed=0,
    ):
        self.time_series = time_series
        self.coords_present = coords_present
        self.coords_future = coords_future
        self.num_variables, self.periods = self._check_and_initialize_args(periods)
        self.seed = seed
        self.imbs_training = None
        self.weights_training = None
        self.weights_final = None
        self.imbs_final = None
        self.errors_final = None
        self.adj_matrix = None
        self.groups_dictionary = None

    def _check_and_initialize_args(self, periods):
        num_variables = None
        periods = periods
        if self.time_series is None and self.coords_present is not None and self.coords_future is not None:
            assert len(self.coords_future.shape) == 3, (
                f"Error: coords_future has shape {self.coords_future.shape}, while the expected shape is (N_samples, D_features, n_lags).\n"
                +"If you want to test a single time lag, provide as input coords_future[:,:,np.newaxis]."
            )
            assert self.coords_present.shape == self.coords_future.shape[:2], (
                "Error: arguments coords_present and coords_future should have shapes (N_samples, D_features) and (N_samples, D_features, n_lags),\n "
                +"but the number of samples and/or the number of features do not match."
            )
            num_variables = self.coords_present.shape[1]
            if periods is not None:
                periods = np.ones(self.coords_present.shape[1]) * np.array(periods)
        elif self.time_series is not None:
            if self.coords_present is not None or self.coords_future is not None:
                warnings.warn(
                    f"You passed the whole time series as input; the arguments coords_present and coords_future will be ignored"
                )
            num_variables = self.time_series.shape[1]
            if periods is not None:
                periods = np.ones(time_series.shape[1]) * np.array(periods)
        return num_variables, periods

    def return_nn_indices(
        self,
        variables,
        num_samples,
        time_lags,
        embedding_dim=1,
        embedding_time=1,
        discard_close_ind=None,
    ):
        """Returns the indices of the nearest neighbors of each point, given the sampling method.

        Args:
            variables (list, jnp.array(int)): array of the coordinates used to build the distance space (with weights 1)
            num_samples (int): number of samples harvested from the full time series
            time_lags (list(int), np.array(int)): tested time lags between 'present' and 'future'
            embedding_dim (int): dimension of the time-delay embedding vector built on each variable. Default is 1, 
                which means the time-delay embeddings are not employed.
            embedding_time (int): lag between consecutive samples in the time-delay embedding vectors of each
                variable. Default is 1.
            discard_close_ind (int): defines the "close points" for which distances and ranks are not computed: for each point i, 
                the distances d[i,i-discard_close_ind:i+discard_close_ind+1] are discarded.
        Returns:
            nn_indices (np.array(float)): array of the nearest neighbors indices: nn_indices[i] is the index of the column
                with value 1 in the rank matrix
        """
        assert self.time_series is not None, (
            "Error: to call this method, provide the time series while initializing the CausalGraph class. "
        )

        assert num_samples < self.time_series.shape[0]-max(time_lags), (
            f"Error: cannot extract {num_samples} samples from {self.time_series.shape[0]} initial samples, "
            +f"if the maximum time lag is {max(time_lags)}.\nChoose a value of num_samples such that "
            +f"num_samples < {self.time_series.shape[0]} - {max(time_lags)}"
        )
        indices_present = np.linspace((embedding_dim-1) * embedding_time, # select times defining the ensemble of trajectories
                                    self.time_series.shape[0]-max(time_lags)-1, 
                                    num_samples, dtype=int)
        indices_present = [indices_present - embedding_time * i for i in range(embedding_dim)]
        coords_present = self.time_series[indices_present]            # has shape (embedding_dim, num_samples, n_variables)
        coords_present = np.transpose(coords_present, axes=[1, 2, 0]) # convert to shape (num_samples, n_variables, embedding_dim)
        dii = DiffImbalance(
            data_A=coords_present[:,variables].reshape((num_samples, len(variables) * embedding_dim)),
            data_B=coords_present[:,variables].reshape((num_samples, len(variables) * embedding_dim)), # dummy argument
            discard_close_ind=discard_close_ind
        )
        nn_indices = dii._return_nn_indices() #variables=variables)
        return np.array(nn_indices)

    def optimize_present_to_future(
        self,
        num_samples,
        time_lags,
        embedding_dim_present=1,
        embedding_dim_future=1,
        embedding_time=1,
        target_variables="all",
        save_weights=False,
        dii_threshold=0.8,
        num_epochs=100,
        batches_per_epoch=1,
        batches_method="all_columns",
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
        discard_close_ind=None,
    ):
        """Iteratively optimizes the DII from the full space in the present to a target space in the future.
    
        Args:
            num_samples (int): number of samples harvested from the full time series, interpreted as
                independent initial conditions of the same dynamical process
            time_lags (list(int), np.ndarray(int)): tested time lags between 'present' and 'future'
            embedding_dim_present (int): dimension of the time-delay embedding vectors built in the optimized 
                space (t=0, t=-1, ...). Default is 1, which means the time-delay embeddings are not employed.
            embedding_dim_future (int): dimension of the time-delay embedding vectors built in the space of 
                the target variable (t=tau, t=tau-1, ...). Default is 1.
            embedding_time (int): lag between consecutive samples in the time-delay embedding vectors of each
                variable.  Default is 1.
            target_variables (str or list(int), np.array(int)): list or np.array of the target variables
                defining the distance space in the future. By default target_variables=="all", which means 
                that the optimization is iterated over all the variables as target.
            num_epochs (int): number of training epochs
            batches_per_epoch (int): number of minibatches; must be a divisor of n_points. Each update of the weights is
                carried out by computing the gradient over n_points / batches_per_epoch points. The default is 1, which
                means that the gradient is computed over all the available points (batch GD).
            batches_method (str): method for minibatch implementation (either 'all_columns' or 'sample_columns')
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
            save_weights (bool): whether to save or not the weights during each training. If True, weights are saved
                in the argument 'weights_training' of the CausalGraph object, which is an array of shape 
                (n_target_variables, n_time_lags, num_epochs+1, num_variables).

        Returns:
            weights_final (np.array(float)): array of shape (n_target_variables,n_time_lags,D) containing the
                D final scaling weights for each optimization, where D is the number of variables in the time series
            imbs_training (np.array(float)): array of shape (n_target_variables, n_time_lags, num_epochs+1) containing the DII 
                during the whole trainings
            imbs_final (np.array(float)): array of shape (n_target_variables, n_time_lags) containing the DII at the end of
                each training, computed over the full data set even when mini-batches are used in the training
            errors_final (np.array(float)): array of shape (n_target_variables, n_time_lags) containing the errors of the DII 
                at the end of each training, computed over the full data set. When compute_error == False, the array only contains
                'None' elements
        """
        if self.time_series is not None:
            assert num_samples < self.time_series.shape[0]-max(time_lags), (
                f"Error: cannot extract {num_samples} samples from {self.time_series.shape[0]} initial samples, "
                +f"if the maximum time lag is {np.max(time_lags)}.\nChoose a smaller value of num_samples."
            )
            indices_present = np.linspace((embedding_dim_present-1) * embedding_time, # select times defining the ensemble of trajectories
                                        self.time_series.shape[0]-max(time_lags)-1, 
                                        num_samples, dtype=int)
            indices_present = np.array([indices_present - embedding_time * i for i in range(embedding_dim_present)])
            coords_present = self.time_series[indices_present]            # has shape (embedding_dim_present, num_samples, n_variables)
            coords_present = np.transpose(coords_present, axes=[1, 2, 0]) # convert to shape (num_samples, n_variables, embedding_dim_present)
            coords_present = coords_present.reshape((num_samples, self.num_variables * embedding_dim_present))
        elif self.coords_present is not None:
            assert time_lags is None or len(time_lags) == self.coords_future.shape[2], (
                f"Error: time_lags contains {len(time_lags)} elements but the last axis of coords_future has "
                +f"dimension {self.coords_future.shape[2]}.\nProvide the correct number of time lags or set time_lags=None."
            )
            if time_lags is None:
                time_lags = np.arange(1,self.coords_future.shape[2]+1)
            coords_present = self.coords_present
        else:
            print("Error: to call this method, provide the time series or the present and future coordinates while initializing the CausalGraph class.")

        if target_variables == "all":
            target_variables = np.arange(self.num_variables)

        imbs_training = np.zeros((len(target_variables),len(time_lags),num_epochs+1))
        if embedding_dim_present == 1:
            weights_final = np.zeros((len(target_variables),len(time_lags),self.num_variables))
            if save_weights is True:
                weights_training = np.zeros((len(target_variables),len(time_lags),num_epochs+1,self.num_variables))
        elif embedding_dim_present > 1:
            weights_final = np.zeros((len(target_variables),len(time_lags),self.num_variables,embedding_dim_present))
            if save_weights is True: 
                weights_training = np.zeros((len(target_variables),len(time_lags),num_epochs+1,self.num_variables,embedding_dim_present))
        imbs_final = np.zeros((len(target_variables),len(time_lags)))
        errors_final = np.zeros((len(target_variables),len(time_lags)))
        # loop over target variables and time lags
        for i_var, target_var in enumerate(target_variables):
            for j_tau, tau in enumerate(time_lags):
                indices_future = np.linspace((embedding_dim_future-1) * embedding_time, # select times defining the ensemble of trajectories
                                        self.time_series.shape[0]-max(time_lags)-1, 
                                        num_samples, dtype=int)
                indices_future = np.array([indices_future - embedding_time * i for i in range(embedding_dim_future)]) + tau

                if self.time_series is not None:
                    coords_future = self.time_series[indices_future, target_var] # has shape (embedding_dim_future, num_samples)
                    coords_future = np.transpose(coords_future, axes=[1, 0])         # convert to shape (num_samples, embedding_dim_future)
                else:
                    coords_future = self.coords_future[:,:,j_tau]

                dii = DiffImbalance(
                    data_A=coords_present,
                    data_B=coords_future,
                    periods_A=self.periods, 
                    periods_B=None if self.periods is None else self.periods[target_var], 
                    seed=self.seed,
                    num_epochs=num_epochs,
                    batches_per_epoch=batches_per_epoch,
                    batches_method=batches_method,
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
                weights_temp, imbs_training[i_var,j_tau] = dii.train(bar_label=f"target_var={target_var}, tau={tau}")
                # save final weights
                #weights_final_temp = weights_temp[-1].reshape((self.num_variables, embedding_dim_present))
                #weights_final_temp = np.max(weights_final_temp, axis=1) # for each variable take only largest weight over embedding components
                #weights_final[i_var,j_tau] = weights_final_temp
                # save final imbalance (on the full dataset) and its error
                imbs_final[i_var,j_tau] = dii.imb_final
                errors_final[i_var,j_tau] = dii.error_final

                # save weights
                if embedding_dim_present == 1:
                    weights_final[i_var,j_tau] = weights_temp[-1].reshape((self.num_variables))
                    if save_weights is True: 
                        weights_training[i_var,j_tau] = weights_temp.reshape((num_epochs+1, self.num_variables))
                elif embedding_dim_present > 1:
                    weights_final[i_var,j_tau] = weights_temp[-1].reshape((self.num_variables, embedding_dim_present))
                    if save_weights is True:
                        weights_training[i_var,j_tau] = weights_temp.reshape((num_epochs+1, self.num_variables, embedding_dim_present))
                

                if imbs_final[i_var,j_tau] > dii_threshold:
                    print(f"The final DII is {imbs_final[i_var,j_tau]:2f}. Discard this and larger time lags for reliable results.")
        
        self.weights_final = weights_final
        self.imbs_training = imbs_training
        if save_weights:
            self.weights_training = weights_training
        self.imbs_final = imbs_final
        self.errors_final = errors_final
        return weights_final, imbs_training, imbs_final, errors_final
        
    def compute_adj_matrix(self, weights, threshold=1e-1):
        """Computes the adjacency matrix from the optimized weights produced by the method optimize_present_to_future.

        As a preliminary step before applying the threshold, the maximum weight over the tested time lags is taken
        for each pair X_i(0) -> X_j(tau) (i,j=1,...,D)

        Args:
            weights (np.ndarray(float)): array of shape (D,n_time_lags,D) containing the optimal scaling weights 
                produced by optimize_present_to_future with the option target_variables="all"
            threshold (float): value of the threshold used to construct the adjacency matrix. If a weight is smaller
                than the threshold the corresponding entry in the adjacency matrix is set to 0, otherwise it is set to 1.

        Returns:
            adj_matrix (np.ndarray(float)): array of shape (D,D) defining the adjacency matrix of a directed graph, where
                each arrow defines a direct or indirect link
        """
        assert weights is not None, (
            "Error: to call this method, provide the weights obtained with the method optimize_present_to_future, "
            +"with the option target_variables='all'"
        )
        assert len(weights.shape) == 3, (
            "Error: the array of weight must have shape (D,n_time_lags,D). If you are testing a single time lag, reshape "
            +"this array with weights[:,np.newaxis,:]."
        )
        assert weights.shape[0] == weights.shape[2], (
            "Error: the array of weight must have shape (D,n_time_lags,D), where D is the number of variables"
        )
        weights_max = np.max(weights, axis=1) # select maximum weights over all tested time lags
        adj_matrix = np.zeros((weights.shape[0], weights.shape[2])) 
        adj_matrix[weights_max.T > threshold] = 1 # apply threshold
        self.adj_matrix = adj_matrix
        return adj_matrix
    
    def _ancestors(self, adj_matrix=None): # this replaces function "ancestors" coded by Matteo
        """Finds the ancestors of each node in the directed graph described by the input adjacency matrix.

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
        """Finds the groups of variables defining new nodes in the coarse-grained causal graph.

        Args:
            adj_matrix (np.ndarray(float)): binary matrix of shape (D,D) defining the links of a directed 
                graph with D nodes.

        Returns:
            groups_dictionary (dict): dictionary with pairs (group_id, order) as keys and lists containing
                the indices of the variables in each group as values. group_id is an integer number identifying
                the group, while order is an integer identifying the step of the algorithm at which the group is
                identified, namely its level of autonomy. The keys are sorted from the smallest to the largest order. 
                Both group_id and order start from 0.
        """
        assert adj_matrix is not None, (
            "Error: provide as intput the adjacency matrix computed with the method compute_adj_matrix"
        )
        
        min_auto_sets = self._ancestors(adj_matrix)
        # re-order minimal autonomous sets from smallest to largest size
        sizes_auto_sets = [len(auto_set) for auto_set in min_auto_sets]
        min_auto_sets = [min_auto_sets[i_sorted] for i_sorted in np.argsort(sizes_auto_sets)]

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


    def community_graph_visualization(self, groups_dictionary, adj_matrix, type=None, **kwargs): 
        """Plots a visual representation of the coarse-grained causal graph

        Args:
            groups_dictionary (dict): dictionary with pairs (group_id,level) as keys and lists containing
                the indices of the variables in each group as values
            adj_matrix (np.array(float)): matrix of shape (D,D) defining the links between the variables 
                after thresholding the matrix of the optimized weights
            type (str): if "microscopic" it returns the visualization with one node for each variable

        Returns:
            G (nx.diGraph object): final coarse-grained causal graph
        """
        assert adj_matrix is not None, (
            "Error: provide as intput the adjacency matrix computed with the method compute_adj_matrix"
        )
        assert groups_dictionary is not None, (
            "Error: provide as intput the groups dictionary computed with the method find_groups"
        )

        if type == 'microscopic':   
            G_=nx.from_numpy_array(adj_matrix,create_using=nx.DiGraph)

            features={}
            metafeatures={}

            for element in groups_dictionary.items():
                for variable in element[1]:
                    features.update({variable:element[0][0]})
                    metafeatures.update({variable:element[0][1]})

            communities=[set([el for el, pos in features.items() if pos==k ]) for k in set(features.values())]
            metacommunities=[set([el for el, pos in metafeatures.items() if pos==k ]) for k in set(metafeatures.values())]

            assert len(communities) != 1, (
            f'Error: Only one group is present. Try plotting with a standard function of networkx.')
            
            G=nx.DiGraph()
            for comm in communities:
                G.add_node(str(list(comm)))
            for i in range(len(communities)):
                present=list(communities[i])
                time = metafeatures[present[0]]
                if time < len(metacommunities) - 1:
                    for step in range(1,len(metacommunities) - time):
                        future = list(metacommunities[metafeatures[list(communities[i])[0]]+step])
                        connections = (np.where(adj_matrix[np.ix_(present,future)]!=0))
                        for j in range(len(connections[0])):
                            looking = future[connections[1][j]]
                            final = communities[np.where([looking in communities[i] for i in range(len(communities))])[0][0]]
                            G.add_edge(str(present),str(list(final)))

            iter = [el for el in G.edges]
            for edge in iter:
                if sum(1 for _ in nx.all_simple_paths(G, source=edge[0], target=edge[1]))>1:
                    G.remove_edge(edge[0],edge[1])


            options = {'scale':0.1, 'k1':1, 'k2':2, 'cmap':plt.cm.Blues, 
            }
            options.update(kwargs)


            # Compute positions for the node clusters as if they were themselves nodes in a
            # supergraph using a larger scale factor
            superpos = nx.spring_layout(G, k=options['k1'], seed=429)

            # Use the "supernode" positions as the center of each node cluster
            centers = list(superpos.values())
            pos = {}
            for center, comm in zip(centers, communities):
                pos.update(nx.spring_layout(nx.subgraph(G_, comm), scale=options['scale'], k=options['k2'], center=center, seed=1430))

            nx.draw(G_, pos=pos, node_color=[metafeatures[i] for i in range(len(adj_matrix))], cmap=plt.cm.Blues, with_labels=True)
            plt.show()
            return G

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
        groups_orders = {key[1]: [] for key in keys}
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
        options.update(kwargs)
        nx.draw_circular(G, arrows=True, with_labels=True, **options)
        plt.show()

        # return networkx object
        return G