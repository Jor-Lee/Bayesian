import os
import time
import sys
import pandas as pd
import heapq
import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle

from scipy.spatial.distance import cdist
from scipy.stats import norm

import tensorflow as tf

class BO:
    """
    A class for Bayesian Optimization (BO) to find the optimal input values that maximise 
    or minimise an objective function.

    Parameters
    ----------
    KernelFunction : callable
        Function representing the kernel used in the Gaussian Process.
    length_scale : float
        The length scale parameter of the kernel.
    bounds : array-like, shape (n_dimensions, 2)
        Array representing the bounds of each dimension.
    AcquisitionFunction : callable
        Function to calculate the acquisition value.
    acquisition_samples : int
        Number of candidate points to sample for the acquisition function.
    random_seed : int, optional
        Seed for random number generation to ensure reproducibility.
    minimise : bool, optional
        If True, the objective is to minimise the function instead of maximizing it.
    log_path : str, optional
        Path to the log file for logging the optimization process. No logging is done if log_path is None.
    dynamic_bounds : bool, optional
        If True, dynamically reduce the search bounds based on optimization progress to focus on promising areas.
    iterations_between_reducing_bounds : int, optional
        Number of iterations without improvement required before the bounds are reduced. Used only if dynamic_bounds is True.
    first_reduce_bounds : int, optional
        Minimum number of iterations before starting to reduce bounds. Used only if dynamic_bounds is True.
    reduce_bounds_factor : float, optional
        Factor by which to reduce bounds. Used only if dynamic_bounds is True.
    """

    def __init__(self, PriorMeanFunction, KernelFunction, length_scale, bounds, AcquisitionFunction, acquisition_samples, noise=1e-5, random_seed=None, minimise=False, log_path=None, dynamic_bounds=False, iterations_between_reducing_bounds=None, first_reduce_bounds=None, reduce_bounds_factor=None):
        # initialise class attributes with provided parameters.
        self.PriorMean = PriorMeanFunction
        self.Kernel = KernelFunction
        self.length_scale = length_scale
        self.bounds = bounds
        self.noise = max(noise, 1e-5)
        
        self.AcquisitionFunction = AcquisitionFunction
        self.acquisition_samples = acquisition_samples

        self.random_seed = random_seed
        self.minimise = minimise

        self.log_path = log_path

        self.iteration_number = 0
        self.iterations_array = np.empty([0, 1])

        self.X_data = self.X_data = np.empty([0, bounds.shape[0]])
        self.y_data = np.empty([0, 1])

        # Set the random seed for reproducibility
        if random_seed is not None:
            np.random.seed(self.random_seed)

        if self.log_path is not None:
            self.CreateLogger()

        # Dynamic bounds configuration
        self.dynamic_bounds = dynamic_bounds
        if self.dynamic_bounds:
            self.iterations_between_reducing_bounds = iterations_between_reducing_bounds
            self.first_reduce_bounds = first_reduce_bounds
            self.reduce_bounds_factor = reduce_bounds_factor

            self.bounds_reduction_counter = 0
            self.stuck_in_peak_counter = 0
            self.current_best_value = -np.inf

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - Get Next X - -- -- -- -------------------================ #

    def GetRandomXBatch(self, batch_size):
        """
        Generate a batch of random X values within the specified bounds.

        This method generates random points within the bounds for each batch of 
        simulations, ensuring that each point is unique within the batch.

        Parameters
        ----------
        batch_size : int
            Number of random points to generate.

        Returns
        -------
        raw_X : ndarray, shape (batch_size, n_dimensions)
            Randomly generated X values for the batch.
        """
        if self.log_path is not None:
            optimiser_start_time = time.time()  # Record start time for optimization
            self.logger.info(f'Getting a batch of random X values.')

        # Generate random points within the bounds for each dimension
        raw_X = np.array([np.array([np.random.uniform(lower_bound, upper_bound) for (lower_bound, upper_bound) in self.bounds]) for i in range(batch_size)])

        if self.log_path is not None:
            optimiser_end_time = time.time()  # Record end time for optimization
            self.logger.debug(f'The time taken to get all X values for the random iteration was {(optimiser_end_time-optimiser_start_time)/60} minutes.\n')
            self.logger.info(f'Got the random batch of X values.')

        if self.dynamic_bounds==True:
            self.batch_size = batch_size   

        return raw_X
    
    def GetNextX(self, kappa='default', K_inv=None):
        """
        Get the next set of input parameters for the objective function.

        This method computes the next best set of input parameters by optimizing the
        acquisition function based on the predicted mean, standard deviation, and kappa value.

        Parameters
        ----------
        kappa : float or str, optional
            Kappa value for the acquisition function. If 'default', use the default kappa for the acquisition function.
        K_inv : ndarray, optional
            Precomputed inverse of the kernel matrix. If None, it will be computed internally.

        Returns
        -------
        next_X : ndarray, shape (1, n_dimensions)
            The next set of input parameters (X).
        """
        if self.X_data.size == 0:
            # If no data has been collected, return a random point within the bounds
            next_X = np.array([np.random.uniform(lower_bound, upper_bound) for (lower_bound, upper_bound) in self.bounds])  
        else:
            if K_inv is None:
                # Compute the inverse of the kernel matrix if not already provided
                K_inv = self.InverseKernel()

            # Generate a set of 'acquisition_samples' candidate points from random samples within the bounds
            candidate_x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.acquisition_samples, self.bounds.shape[0]))

            # Predict the mean and standard deviation at each of these points
            mean, variance = self.PredictMeanVariance(candidate_x, K_inv=K_inv)

            # Calculate the acquisition value for each candidate point
            if kappa == 'default':
                candidate_y = self.AcquisitionFunction(mean, np.sqrt(variance))
            else:
                candidate_y = self.AcquisitionFunction(mean, np.sqrt(variance), kappa)

            # Choose the X value which corresponds to the largest acquisition value
            max_index = np.argmax(candidate_y)
            next_X = np.array([candidate_x[max_index]])

        return next_X
    

    def GetNextXBatch(self, batch_size, sub_batch_size=None, kappa = 'default', max_kappa = None, min_kappa = None):
        """
        Get the next batch of input parameters for the objective function.

        This method generates a batch of input parameters by iteratively optimizing 
        the acquisition function. It accounts for the exploration-exploitation trade-off 
        using kappa values and ensures uniqueness of the points.

        Parameters
        ----------
        batch_size : int
            The number of points to generate in the batch.
        sub_batch_size : int, optional
            Size of sub-batches if sub-batch optimization is used. If None, sub-batch optimization is not used.
        kappa : float or str, optional
            Kappa value for the acquisition function. If 'default', use the default kappa for the acquisition function.
        max_kappa : float, optional
            Maximum kappa value for exponential kappa distribution.
        min_kappa : float, optional
            Minimum kappa value for exponential kappa distribution.

        Returns
        -------
        raw_X : ndarray, shape (batch_size, n_dimensions)
            The next batch of input parameters (X).
        """
        if self.X_data.size == 0:
            if self.log_path is not None:
                self.logger.info('Since there is no data stored in the object, this batch is random.')

            raw_X = self.GetRandomXBatch(batch_size)

        else:
            if self.log_path is not None:
                optimiser_start_time = time.time()  # Record start time for optimization
                self.logger.info(f'Getting X values for this iteration.')

            if sub_batch_size is None:
                raw_X = np.empty((batch_size,len(self.bounds)))  # initialise the list to store the batch of X values
                K_inv = self.InverseKernel()

                for i in range(batch_size):
                    if max_kappa is not None and min_kappa is not None:
                            # Calculate kappa for the current point within the batch
                            kappa = self.CalculateKappa(batch_size, i, max_kappa, min_kappa)
                            raw_X[i] = self.GetNextX(kappa, K_inv=K_inv)[0]
                    else:
                        if kappa == 'default':
                            # Get the next best point with the default kappa value
                            raw_X[i] = self.GetNextX(K_inv=K_inv)[0]
                        else:
                            raw_X[i] = self.GetNextX(kappa, K_inv=K_inv)[0]
                if self.log_path is not None:
                    self.logger.info(f'Got the batch of X values.')

            if sub_batch_size:
                raw_X = np.empty((batch_size,len(self.bounds)))  # initialise the list to store the batch of X values

                for i in range(int(np.ceil(batch_size / sub_batch_size))):
                    if self.log_path is not None: 
                        self.logger.info(f'\nGetting X values for sub-batch {i}.')
                    sub_raw_X = np.empty((sub_batch_size, len(self.bounds)))
                    sub_raw_y = np.empty([sub_batch_size, 1])
                    K_inv = self.InverseKernel()

                    for j in range(sub_batch_size):
                        if max_kappa is not None and min_kappa is not None:
                                kappa = self.CalculateKappa(sub_batch_size, j, max_kappa, min_kappa)
                                sub_raw_X[j] = self.GetNextX(kappa, K_inv=K_inv)[0]
                        else:
                            if kappa == 'default':
                                sub_raw_X[j] = self.GetNextX(K_inv=K_inv)[0]
                            else:
                                sub_raw_X[j] = self.GetNextX(kappa, K_inv=K_inv)[0]

                        raw_X[j + sub_batch_size * i] = sub_raw_X[j]

                    normalised_sub_raw_y = self.PredictMeanVariance(np.array(sub_raw_X), K_inv=K_inv)[0]

                    sub_raw_y = np.min(self.y_data) + ((normalised_sub_raw_y + 1) * (np.max(self.y_data) - np.min(self.y_data)) / 2)

                    if self.minimise is True:
                        sub_raw_y = -sub_raw_y

                    # Concatenate sub-batch data to the existing X_data and y_data arrays
                    self.X_data = np.vstack([self.X_data, sub_raw_X])
                    self.y_data = np.vstack([self.y_data, sub_raw_y])

                    if self.log_path is not None:
                        self.logger.info(f'Got X values for sub-batch {i}.')

                # Remove the initial empty entries
                self.X_data = self.X_data[:-batch_size]
                self.y_data = self.y_data[:-batch_size]

                if self.log_path is not None: 
                    self.logger.info(f'\nX values for all sub-batches have been attained.')

            if self.log_path is not None:
                optimiser_end_time = time.time()  # Record end time for optimization
                self.logger.debug(f'The time taken to get all X values for this iteration was {(optimiser_end_time-optimiser_start_time)/60} minutes.')

        if self.dynamic_bounds is True:
            self.batch_size = batch_size        

        return raw_X

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- - Update object - -- -- -- -------------------================ #

    def UpdateData(self, raw_X, raw_y, update_iteration=True):
        """
        Update the internal data storage with new X and Y values.

        This method appends the new input parameters (X) and output results (Y) from 
        the current iteration to the existing data arrays stored in the class.

        Parameters
        ----------
        raw_X : ndarray, shape (n_samples, n_dimensions)
            The new input parameters to append.
        raw_y : ndarray, shape (n_samples, 1)
            The new output results to append.
        update_iteration : bool, optional
            If True, increment the iteration counter after updating the data.
        """
        # Append new X and Y data to existing data arrays
        if self.X_data.size == 0:
            self.X_data = raw_X
        else:
            self.X_data = np.vstack([self.X_data, raw_X])

        if self.y_data.size == 0:
            self.y_data = raw_y
        else:
            self.y_data = np.vstack([self.y_data, raw_y])

        if update_iteration:
            if self.log_path is not None:
                self.logger.info(f'Data has been updated for iteration number {self.iteration_number}')

            # Increment iteration count and update iteration array
            self.iterations_array = np.vstack([self.iterations_array, np.full((len(raw_y), 1), self.iteration_number)])
            self.iteration_number += 1

        if self.dynamic_bounds==True:
            # Check if optimization meets the bounds' reduction criteria and update bounds if necessary
            self.CheckImprovement()
            self.UpdateBounds()

        if self.log_path is not None:
            self.LogCurrentStatus()



    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - - Algebra - - -- -- -- -------------------================ #

    def InverseKernel(self):
        """
        Calculate the inverse of the kernel matrix (K_inv) using Cholesky decomposition.

        This method adds Gaussian noise to the model.

        Returns
        -------
        K_inv : ndarray, shape (n_samples, n_samples)
            The inverse of the kernel matrix.
        """
        # Compute the kernel matrix with the Gaussian noise
        K = self.Kernel(self.X_data, self.X_data, self.length_scale) + self.noise * np.eye(len(self.X_data))

        # Cholesky decomposition: K = L L^T
        try:
            L = np.linalg.cholesky(K)
            
            # Solve for the inverse using the decomposition
            L_inv = np.linalg.solve(L, np.eye(L.shape[0]))  # Solving L * L_inv = I
            K_inv = L_inv.T @ L_inv  # Compute K_inv = (L_inv.T) * L_inv
        except np.linalg.LinAlgError:
            print("Cholesky decomposition failed, falling back to np.linalg.pinv (pseudoinverse).")
            K_inv = np.linalg.pinv(K)  # Compute pseudoinverse as fallback

        # End inversion timer

        return K_inv




    def PredictMeanVariance(self, candidate_x, K_inv=None, normalised=True, batch_size=1000):
        """
        Predict the mean and variance using GPU-accelerated matrix operations with TensorFlow.
        Processes candidate_x in batches.

        Parameters:
        - candidate_x: NumPy array of shape (n_candidates, d)
        - K_inv: (n_train, n_train) NumPy array, inverse kernel matrix (precomputed)
        - normalised: If True, returns normalised values
        - batch_size: Batch size for kernel computations

        Returns:
        - mean: NumPy array of shape (n_candidates, 1)
        - variance: NumPy array of shape (n_candidates, 1)
        """

        # Choose the device (GPU if available, else CPU)
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

        with tf.device(device):
            if K_inv is None:
                K_inv = self.InverseKernel()  # Compute the inverse kernel matrix if not provided
            
            K_inv = tf.convert_to_tensor(K_inv, dtype=tf.float64)
            candidate_x = tf.convert_to_tensor(candidate_x, dtype=tf.float64)

            prior_data_mean = tf.convert_to_tensor(self.PriorMean(self.X_data), dtype=tf.float64)

            # Normalize y_data (on the same device)
            y_min, y_max = tf.reduce_min(self.y_data), tf.reduce_max(self.y_data)

            if y_max - y_min != 0.0:
                normalised_y_data = 2 * (self.y_data - y_min) / (y_max - y_min) - 1
            else:
                normalised_y_data = tf.zeros_like(self.y_data)  # Handle edge case where y_max == y_min

            if self.minimise:
                normalised_y_data = -normalised_y_data

            normalised_y_data = tf.convert_to_tensor(normalised_y_data, dtype=tf.float64)

            # Initialize lists to hold batch results
            mean_list = []
            var_list = []

            # Process candidate_x in batches
            n_candidates = candidate_x.shape[0]

            for i in range(0, n_candidates, batch_size):
                # Select batch
                candidate_x_batch = candidate_x[i:i + batch_size]

                prior_candidate_mean_batch = self.PriorMean(candidate_x_batch)

                # Compute kernel matrices for the batch
                K_star = tf.convert_to_tensor(self.Kernel(self.X_data, candidate_x_batch, self.length_scale), dtype=tf.float64)
                K_star_star = tf.convert_to_tensor(self.Kernel(candidate_x_batch, candidate_x_batch, self.length_scale), dtype=tf.float64)

                # Compute the mean using matrix multiplication (GPU-accelerated)
                mean_batch = tf.convert_to_tensor(prior_candidate_mean_batch, dtype=tf.float64) + tf.matmul(tf.matmul(K_star, K_inv, transpose_a=True), (normalised_y_data - prior_data_mean))

                # Compute the full covariance matrix of the prediction
                cov_reduction = tf.matmul(tf.matmul(K_star, K_inv, transpose_a=True), K_star)
                var_batch = tf.linalg.diag_part(K_star_star - cov_reduction)

                # Store batch results
                mean_list.append(mean_batch)
                var_list.append(var_batch)

            # Concatenate the results from all batches
            mean = tf.concat(mean_list, axis=0) 
            var = tf.concat(var_list, axis=0)

            # Convert results back to NumPy if needed
            mean, var = mean.numpy(), var.numpy()

            mean = mean.reshape(n_candidates, 1)
            var = var.reshape(n_candidates, 1)

            # Optionally return normalised values
            if normalised:
                return mean, var

            # If not normalised, convert back to the original scale
            actual_mean = y_min + (y_max - y_min) * (mean + 1) / 2
            var_scaling_factor = ((y_max - y_min) / 2) ** 2 if y_max - y_min != 0 else (abs(y_max) / 2) ** 2
            actual_var = var * var_scaling_factor

            return actual_mean, actual_var


    
    def CalculateKappa(self, batch_size, current_simulation_number, max_kappa, min_kappa):
        """
        Compute the exploration-exploitation trade-off parameter kappa for the current batch.

        Kappa determines the exploration-exploitation trade-off in Bayesian optimization. 
        It is dynamically calculated based on the batch size and the current simulation number.

        Parameters
        ----------
        batch_size : int
            The number of points in the current batch.
        current_simulation_number : int
            The index of the current simulation within the batch.
        max_kappa : float
            The maximum kappa value for exponential distribution.
        min_kappa : float
            The minimum kappa value for exponential distribution.

        Returns
        -------
        kappa : float
            The computed kappa value for the current simulation.
        """
        try:
            # Calculate kappa using exponential distribution for exploration-exploitation trade-off
            if batch_size == 1:
                kappa = (max_kappa - min_kappa) / 2
            else:
                b = 1/(batch_size-1) * np.log(max_kappa / min_kappa)
                kappa = min_kappa * np.exp(b*current_simulation_number)
        except Exception as e:
            if self.log_path is not None:
                self.logger.error(f"An error occurred: {e}")   

        return kappa

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                                
    # ==============----------------- -- -- - Bounds Reduction - -- -- -------------------================ #   

    def CheckImprovement(self):
        """
        Check if the optimiser has made an improvement.

        This method compares the best value from the most recent batch of iterations 
        with the current best value recorded. If an improvement is found (i.e., a new 
        best value is discovered), it resets the counter that tracks how long the optimiser 
        has been without improvement. If no improvement is observed, the counter is incremented 
        to potentially trigger bounds reduction.
        """
        # Define the range of indices for the most recent batch of data
        relevant_indices = range(max(len(self.y_data) - self.batch_size, 0), len(self.y_data))
        
        if self.minimise is False:
            # Find the index of the largest value in the most recent batch if maximizing
            best_index = heapq.nlargest(1, relevant_indices, key=self.y_data.__getitem__)
        else:
        # Find the index of the smallest value in the most recent batch if minimizing
                best_index = heapq.nsmallest(1, relevant_indices, key=self.y_data.__getitem__)
        
        # Retrieve the best Y value from the identified index
        best_value = [self.y_data[i] for i in best_index]

        # Check if the best value from the recent batch improves upon the current best value
        if (self.minimise is False and best_value[0] > self.current_best_value) or (self.minimise is True and best_value[0] < self.current_best_value):
            # Improvement found, reset the no improvement flag and counter
            self.no_improvement_flag = 0  
            self.no_improvement_counter = 0 
            # Update the current best value to the new best value found
            self.current_best_value = best_value[0] 
        else:
            # No improvement found, set the flag and increment the counter
            self.no_improvement_flag = 1  
            self.no_improvement_counter += 1 

            if self.log_path is not None:
                self.logger.info('The Optimiser has not found a better point')
    
    def UpdateBounds(self):
        """
        Reduce the search bounds if no improvement is found.

        This method reduces the search bounds if the optimiser has not found a better 
        point for a specified number of iterations and if the current iteration number 
        exceeds a certain threshold. It proportionally reduces the bounds around the best 
        observed point and scales the length scale used in the kernel function.
        """
        if self.log_path is not None:
            self.logger.info(f'The optimiser has not found a better point for {self.no_improvement_counter} iterations')

        # Check if bounds should be reduced based on no improvement and iteration thresholds
        if self.no_improvement_counter >= self.iterations_between_reducing_bounds and self.iteration_number >= self.first_reduce_bounds:
            self.no_improvement_counter = 0  # Reset the no_improvement_counter
            self.bounds_reduction_counter += 1  # Increment the bounds_reduction_counter
            self.length_scale = self.length_scale * self.reduce_bounds_factor  # Reduce the length scale proportionally to the bounds
            
            # Find the best observed point to center the new bounds
            best_point = self.X_data[self.BestData()[0][0]]

            # initialise arrays for new bounds and ranges
            new_bounds = np.empty_like(self.bounds, dtype=float)
            new_range = np.empty_like(best_point, dtype=float)

            # Calculate new bounds centered around the best observed point for each dimension
            for i in range(len(self.bounds)):
                new_range[i] = (self.bounds[i,1] - self.bounds[i,0]) * self.reduce_bounds_factor
                new_bounds[i,1] = best_point[i] + new_range[i]/2
                new_bounds[i,0] = best_point[i] - new_range[i]/2

                # Ensure new bounds do not exceed the original bounds
                if new_bounds[i,0] < self.bounds[i,0]:
                    new_bounds[i,0] = self.bounds[i,0]   
                    new_bounds[i,1] = self.bounds[i,0] + new_range[i]  
                elif new_bounds[i,1] > self.bounds[i,1]:
                    new_bounds[i,0] = self.bounds[i,1]-new_range[i] 
                    new_bounds[i,1] = self.bounds[i,1] 

            # Update the bounds with the new values
            self.bounds = new_bounds

            if self.log_path is not None:
                self.logger.info(f'New bounds are {self.bounds}.')

    def BestData(self, number_indices=1):
        """
        Find the best data points in terms of the maximum observed values.

        This method identifies the largest observed values from the optimization process,
        sorts them, and returns the sorted indices and values.

        Parameters
        ----------
        number_indices : int, optional
            The number of top data points to retrieve.

        Returns
        -------
        tuple : (ndarray, ndarray)
            A tuple containing the sorted indices and corresponding values of the best data points.
        """
        if self.minimise is False:
            # Find the indices of the best y values
            best_indices = heapq.nlargest(number_indices, range(len(self.y_data)), key=self.y_data.__getitem__)
        else:
            best_indices = heapq.nsmallest(number_indices, range(len(self.y_data)), key=self.y_data.__getitem__)
        
        # Retrieve the Y values for these indices
        best_values = np.array([self.y_data[i] for i in best_indices])

        # Sort the indices and values into value order
        sorted_indices_and_values = sorted(zip(best_indices, best_values), key=lambda x: x[1], reverse=True)

        # Unzip the sorted pairs back into separate lists of indices and values
        sorted_indices, sorted_values = zip(*sorted_indices_and_values)

        return(np.array(sorted_indices), np.array(sorted_values).flatten())
    

    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #

    # ==============----------------- -- -- -- - CSV Data - - -- -- -- -------------------================ #

    def UpdateDataCSV(self, csv_file, update_iteration=False):
        """
        Update the internal data storage by reading values from a CSV file.

        This method reads a CSV file containing results from previous iterations 
        and updates the internal data storage (X_data and y_data) with the information.

        Parameters
        ----------
        csv_file : str
            The path to the CSV file containing previous results.
        update_iteration : bool, optional
            If True, increment the iteration counter after updating the data.

        Returns
        -------
        raw_X : ndarray, shape (n_samples, n_dimensions)
            The new input parameters read from the CSV file.
        raw_y : ndarray, shape (n_samples, 1)
            The new output results read from the CSV file.
        """
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        raw_y = np.array(df['Result'].values).reshape(len(df['Result'].values),1)
        self.y_data = np.vstack([self.y_data, raw_y])
        csv_data_length = len(df['Result'].values)

        self.X_data = np.vstack( [self.X_data, np.zeros((csv_data_length, len(self.bounds)))] )
        raw_X = np.zeros((csv_data_length, len(self.bounds)))

        # Update X_data and y_data arrays with data from CSV
        for i in range(csv_data_length):
            for k in range(len(self.bounds)):
                self.X_data[len(self.y_data) - csv_data_length +i][k] = df[f'X{k}'][i]
                raw_X[i,k] = df[f'X{k}'][i]

        # Update the iteration count
        if update_iteration:
            if self.log_path is not None:
                self.logger.info(f'Data has been updated from the csv {csv_file} for iteration number {self.iteration_number}')
                self.logger.info('')

            self.iteration_number += 1

        return raw_X, raw_y
    
    
    
    def WriteOutputToCSV(self, csv_path, raw_X, raw_y):
        """
        Write the simulation results to a CSV file.

        This method saves the results of the current iteration, including the input 
        parameters (X) and output results (Y), to a CSV file. If the file does not exist, 
        it creates a new one with headers. Otherwise, it appends the new data to the existing file.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file where results should be written.
        raw_X : ndarray, shape (n_samples, n_dimensions)
            The input parameters used in the current iteration.
        raw_y : ndarray, shape (n_samples, 1)
            The output results corresponding to the input parameters.
        """
        # Create arrays for iteration numbers and simulation numbers
        iteration_numbers = np.full(len(raw_X), self.iteration_number)
        simulation_numbers = range(0, len(raw_X))

        # Create a dictionary to hold the data with column names
        data = {
            'Iteration': np.array(iteration_numbers),
            'Simulation': np.array(simulation_numbers),
            'Result': raw_y[:, 0],
        }

        # Add raw_X values with column names (X0, X1, X2, ...)
        for i in range(np.shape(raw_X)[1]):
            data[f'X{i}'] = raw_X[:,i]

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(data)

        # Check if the CSV file exists, if not, create it and write the headers
        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            # Append new data to the existing CSV file
            df.to_csv(csv_path, mode='a', header=False, index=False)

        if self.log_path is not None:
            self.logger.info('csv file updated.')



    # ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                                
    # ==============----------------- -- -- -- - - Logging -- -- -- -- -------------------================ #            

    def CreateLogger(self):
        """
        Create a logger for the optimization process.

        This function checks if a log file already exists. If it does, the program exits
        to prevent overwriting. If not, a new logger is created and configured.

        Raises
        ------
        SystemExit
            If the log file already exists.
        """
        # Check if the log file exists
        if os.path.exists(self.log_path):
            print(f"Error: The log file at {self.log_path} already exists. Quitting the experiment.")
            sys.exit(1)  # Exit to prevent overwriting the log file

        # Setup logger and set level to INFO
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Setup Log_handler - set mode to 'w' to write
        log_handler = logging.FileHandler(self.log_path, mode='w')
        log_handler.setLevel(logging.INFO)

        # Define the log format (preamble before your message is displayed)
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(log_format)

        # Add the handler to the logger object so you can start writing to the log file
        self.logger.addHandler(log_handler)

        self.logger.info('The log has been created.')
        self.logger.info('')


    def LogCurrentStatus(self):
        """
        Log the current status of the optimization process.

        This method logs the best value found so far, the corresponding X values, 
        the number of random X values used, and how many times the bounds have been reduced.
        """
        self.logger.info(f'Current best y value is {self.BestData()[1][0]}; the corresponding X values are {self.X_data[self.BestData()[0][0]]}')
        if self.dynamic_bounds is True:
            self.logger.info(f'The bounds have been reduced {self.bounds_reduction_counter} times')
        self.logger.info('')
        self.logger.info('')

    def PrintCurrentStatus(self):
        """
        Print the current status of the optimization process.

        This method prints the best value found so far, the corresponding X values, 
        and the number of times the bounds have been reduced.
        """
        print(f'Current best y value is {self.BestData()[1][0]}; the corresponding X values are {self.X_data[self.BestData()[0][0]]}.')
        if self.dynamic_bounds is True:
            print(f'The bounds have been reduced {self.bounds_reduction_counter} times.')



# ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                            
# ==============----------------- -- -- -- - - Kernels - - -- -- -- -------------------================ #         


def RBF_Kernel_GPU(X1, X2, length_scales, batch_size=50000):
    """
    Compute the ARD RBF kernel matrix between two sets of vectors X1 and X2 in batches using TensorFlow.

    Parameters:
    - X1: TensorFlow tensor of shape (n_samples_X1, n_features)
    - X2: TensorFlow tensor of shape (n_samples_X2, n_features)
    - length_scales: TensorFlow tensor of shape (d, 1) (ARD) or a single float (isotropic).
    - batch_size: Number of samples to process in each batch.

    Returns:
    - K_numpy: The RBF kernel matrix as a NumPy array.
    """
    # Choose the device (GPU if available, else CPU)
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    with tf.device(device):
        # Ensure length_scales is an array
        if np.isscalar(length_scales):  # If a single float is given, assume isotropic case
            length_scales = np.full(X1.shape[1], length_scales)  # Shape (d,)
        else:
            length_scales = np.asarray(length_scales)
            if length_scales.shape != (X1.shape[1], 1):
                raise ValueError(f'Length scales must have shape ({X1.shape[1],}), but got {length_scales.shape}.')

        length_scales = tf.convert_to_tensor(length_scales, dtype=tf.float64)

        n_X, d = X1.shape
        n_Y = X2.shape[0]

        X1_scaled = tf.convert_to_tensor(X1, dtype=tf.float64) / tf.transpose(length_scales)
        X2_scaled = tf.convert_to_tensor(X2, dtype=tf.float64) / tf.transpose(length_scales)

        # Initialize the kernel matrix
        K_list = []

        # Process in batches
        for i in range(0, n_X, batch_size):
            X1_batch = X1_scaled[i:i+batch_size]  # Shape (batch_size, d)

            # Compute the squared Euclidean distances between X1_batch and X2
            sq_dists = tf.reduce_sum(tf.square(X1_batch), axis=1, keepdims=True) + \
                    tf.reduce_sum(tf.square(X2_scaled), axis=1) - \
                    2 * tf.matmul(X1_batch, X2_scaled, transpose_b=True)

            # Compute the RBF kernel matrix for this batch
            K_batch = tf.exp(-0.5 * sq_dists)

            # Append the batch kernel matrix to the list
            K_list.append(K_batch)

        # Concatenate the batches to form the full kernel matrix
        K_tf = tf.concat(K_list, axis=0)

        # Convert to NumPy
        K_numpy = K_tf.numpy()

        return K_numpy



def RBF_Kernel_CPU(X1, X2, length_scales, batch_size=1000):
    """
    Compute the ARD RBF kernel matrix between two sets of points in batches (CPU version).

    Parameters:
    - X1: array-like, shape (n_X, d)
        The first set of input points (n_X samples, d dimensions).
    - X2: array-like, shape (n_Y, d)
        The second set of input points (n_Y samples, d dimensions).
    - length_scales: float or array-like, shape (d,)
        The length scale for each dimension (ARD) or a single float for isotropic case.
    - batch_size: int, default=1000
        The number of samples to process in each batch.

    Returns:
    - K: array, shape (n_X, n_Y)
        The RBF kernel matrix.
    """
    # Ensure length_scales is an array
    if np.isscalar(length_scales):  # If a single float is given, assume isotropic case
        length_scales = np.full(X1.shape[1], length_scales)  # Shape (d,)
    else:
        length_scales = np.asarray(length_scales)
        if length_scales.shape != (X1.shape[1], 1):
            raise ValueError(f'Length scales must have shape ({X1.shape[1],}), but got {length_scales.shape}.')

    n_X, d = X1.shape
    n_Y = X2.shape[0]

    # Initialize the kernel matrix
    K = np.empty((n_X, n_Y), dtype=np.float64)

    # Process in batches
    for i in range(0, n_X, batch_size):
        X1_batch = X1[i:i+batch_size]  # Shape (batch_size, d)

        # Normalize X1_batch and X2 by length_scales
        X1_scaled = X1_batch / length_scales.T  # Shape (batch_size, d)
        X2_scaled = X2 / length_scales.T      # Shape (n_Y, d)

        # Compute the squared Euclidean distances between X1_batch and X2
        distances = cdist(X1_scaled, X2_scaled, metric='sqeuclidean')  # Shape (batch_size, n_Y)

        # Apply the RBF kernel function
        K[i:i+batch_size, :] = np.exp(-0.5 * distances)

    return K




def MaternKernel(X1, X2, length_scales, nu=5, batch_size=1000):
    """
    Compute the ARD Matern kernel matrix between two sets of points.

    Parameters:
    - X1: np.ndarray, shape (n_X, d)
        The first set of input points (n_X samples, d dimensions).
    - X2: np.ndarray, shape (n_Y, d)
        The second set of input points (n_Y samples, d dimensions).
    - length_scales: float or array-like, shape (d,)
        The length scale for each dimension (ARD) or a single float for isotropic case.
    - nu: float, optional (default=1.5)
        Controls the smoothness of the function. Common values: 0.5, 1.5, 2.5.

    Returns:
    - K: np.ndarray, shape (n_X, n_Y)
        The Matern kernel matrix.
    """

    from scipy.special import kv, gamma

    # Ensure length_scales is an array
    if np.isscalar(length_scales):  # If a single float is given, assume isotropic case
        length_scales = np.full(X1.shape[1], length_scales)  # Shape (d,)
    else:
        length_scales = np.asarray(length_scales)
        if length_scales.shape != (X1.shape[1], 1):
            raise ValueError(f'Length scales must have shape ({X1.shape[1],}), but got {length_scales.shape}.')

    n_X, d = X1.shape
    n_Y = X2.shape[0]

    # Initialize the kernel matrix
    K = np.empty((n_X, n_Y), dtype=np.float64)

    # Process in batches
    for i in range(0, n_X, batch_size):
        X1_batch = X1[i:i+batch_size]  # Shape (batch_size, d)

        # Normalize X1_batch and X2 by length_scales
        X1_scaled = X1_batch / length_scales.T  # Shape (batch_size, d)
        X2_scaled = X2 / length_scales.T      # Shape (n_Y, d)

        # Compute the squared Euclidean distances between X1_batch and X2
        distances = cdist(X1_scaled, X2_scaled, metric='sqeuclidean')  # Shape (batch_size, n_Y)

        # Compute scaled distances
        scaled_dists = np.sqrt(2 * nu) * distances  # Shape (n_X, n_Y)

        # Compute Matern kernel based on nu value
        if nu == 0.5:
            K[i:i+batch_size, :] = np.exp(-scaled_dists)  # Matern 0.5 (Exponential kernel)
        else:
            # Compute the Matern function
            K[i:i+batch_size, :] = (2 ** (1.0 - nu) / gamma(nu)) * (scaled_dists ** nu) * kv(nu, scaled_dists)

            # Ensure numerical stability: Handle NaN values
            K[i:i+batch_size, :][distances == 0] = 1.0  # K(x, x) = 1 for all nu

    return K



# ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                            
# ==============----------------- -- - - Kernel Optimisation - - -- -------------------================ #   


def LogMarginalLikelihood(X_data, Y_data, kernel_function, noise, *kernel_params):
    """
    Compute the log marginal likelihood (LML) for a given kernel function. Usful in predicting kernel hyper-parameters such as lengthscale. 
    Whichever set of hyper-parameters maximise the LML define the kernel which best predicts the observed data. 

    Parameters:
    - X_data: Input data points (N x d)
    - Y_data: Observed values (N x 1)
    - kernel_function: A function that computes the covariance matrix K(X, X)
    - *kernel_params: Parameters for the kernel function (e.g., length scale, variance, etc.)

    Returns:
    - log marginal likelihood (scalar)
    """
    # Compute the kernel matrix with the provided kernel function
    K = kernel_function(X_data, X_data, *kernel_params) + noise * np.eye(len(X_data))

    # Cholesky decomposition for numerical stability
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_data))  # Equivalent to K_inv @ Y_data

    # Log determinant using Cholesky
    log_det_K = 2 * np.sum(np.log(np.diag(L)))

    # Compute log marginal likelihood
    n = len(Y_data)
    term1 = Y_data.T @ alpha
    term2 = log_det_K
    log_likelihood = -0.5 * term1 - 0.5 * term2 - 0.5 * n * np.log(2 * np.pi)

    return log_likelihood[0][0]


def LengthScalePenalty(length_scales, lambda_penalty=1.0):
    """
    Penalize large disparities between length scales.
    
    Parameters:
    - kernel_params: List of kernel parameters, including length scales
    - lambda_penalty: The strength of the penalty on disparities between length scales
    
    Returns:
    - The regularization penalty term
    """
    penalty = 0.0
    # Sum of squared differences between length scales
    for i in range(len(length_scales)):
        for j in range(i+1, len(length_scales)):
            penalty += (length_scales[i] - length_scales[j])**2

    return lambda_penalty * penalty


def OptimiseLML(Kernel, X_data, Y_data, noise, initial_params, lambda_penalty=1.0):
    from scipy.optimize import minimize

    def objective(log_params):
        params = np.exp(log_params)  # Exponentiate to transform back from log scale
        
        # Regularization penalty to encourage balanced length scales
        penalty = LengthScalePenalty(params, lambda_penalty)
        
        # Compute the log marginal likelihood
        log_likelihood = -LogMarginalLikelihood(X_data, Y_data, Kernel, noise, params.reshape(-1, 1))
        
        # Add the length scale penalty to the objective function
        return log_likelihood + penalty

    initial_log_params = np.log(initial_params.flatten())
    result = minimize(objective, initial_log_params, method='L-BFGS-B')

    optimal_params = np.exp(result.x)

    # Print results
    print("Optimal parameters:", optimal_params)
    print("Function value at optimum:", result.fun)
    print("Converged:", result.success)
    print("Number of iterations:", result.nit)
    print("Number of function evaluations:", result.nfev)

    return optimal_params


# ==========p====----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                            
# ==============----------------- -- -- Acquisition Functions - -- -------------------================ #    

def UpperConfidenceBound(mean, standard_deviation, kappa=0.1):
    """
    Compute the acquisition value for a given set of parameters using the Upper Confidence Bound (UCB) method.

    The UCB method combines the predicted mean and standard deviation with a kappa value to balance 
    exploration and exploitation in selecting the next point to sample.

    Parameters
    ----------
    mean : np.ndarray
        The predicted mean values of the objective function.
    standard_deviation : np.ndarray
        The predicted standard deviation (uncertainty) of the prediction.
    kappa : float, optional
        A parameter that controls the trade-off between exploration and exploitation.

    Returns
    -------
    np.ndarray
        The acquisition values used to guide the selection of the next sample point.
    """
    # Generate a small random noise to avoid deterministic behavior
    random_numbers = 0.05 * (np.random.rand(len(mean)).reshape(len(mean), 1))
    random_numbers = random_numbers * np.max(standard_deviation)

    # Compute the UCB acquisition function
    ucb = mean + kappa * (standard_deviation + random_numbers)

    return ucb

def ExpectedImprovement(mean, standard_deviation, best_observed=1.0, kappa=0.01):
    """
    Compute the Expected Improvement (EI) acquisition function value for a given set of parameters.

    The EI method calculates the expected improvement over the current best observed value, encouraging
    sampling in regions with high uncertainty or potential improvements.

    Parameters
    ----------
    mean : np.ndarray
        The predicted mean values of the objective function.
    standard_deviation : np.ndarray
        The predicted standard deviation (uncertainty) of the prediction.
    best_observed : float, optional
        The current best observed value of the objective function.
    kappa : float, optional
        Exploration parameter, a small positive value to encourage exploration.

    Returns
    -------
    np.ndarray
        The expected improvement values for each point in the input.
    """
    # Calculate the improvement (mean - best_observed - kappa)
    improvement = mean - best_observed - kappa
    
    # Calculate the Z value for standardization
    Z = improvement / (standard_deviation + 1e-9)  # Adding epsilon to avoid division by zero
    
    # Calculate the Expected Improvement
    ei = improvement * norm.cdf(Z) + standard_deviation * norm.pdf(Z)
    
    # Ensure non-negative EI values
    for i in range(len(ei)):
        ei[i] = np.max(ei[i], 0)

    return ei

def ProbabilityImprovement(mean, standard_deviation, best_observed=1.0, kappa=0.01):
    """
    Compute the Probability of Improvement (PI) acquisition function value for a given set of parameters.

    The PI method calculates the probability that the objective function will improve upon the current best 
    observed value, balancing exploration and exploitation.

    Parameters
    ----------
    mean : np.ndarray
        The predicted mean values of the objective function.
    standard_deviation : np.ndarray
        The predicted standard deviation (uncertainty) of the prediction.
    best_observed : float, optional
        The current best observed value of the objective function.
    kappa : float, optional
        Exploration parameter, a small positive value to encourage exploration.

    Returns
    -------
    np.ndarray
        The probability of improvement values for each point in the input.
    """
    # Calculate the improvement (mean - best_observed - kappa)
    improvement = mean - best_observed - kappa
    
    # Calculate the Z value for standardization
    Z = improvement / (standard_deviation + 1e-9)  # Adding epsilon to avoid division by zero
    
    # Calculate the Probability of Improvement
    pi = norm.cdf(Z)

    return pi

def KnowledgeGradient(mean, standard_deviation, best_observed=1.0, kappa=0.01):
    """
    Compute the Knowledge Gradient (KG) acquisition function value for a given set of parameters.

    The KG method calculates the expected increase in the value of the best solution found so far by sampling 
    a new point, normalizing the expected improvement by the standard deviation.

    Parameters
    ----------
    mean : np.ndarray
        The predicted mean values of the objective function.
    standard_deviation : np.ndarray
        The predicted standard deviation (uncertainty) of the prediction.
    best_observed : float, optional
        The current best observed value of the objective function.
    kappa : float, optional
        Exploration parameter, a small positive value to encourage exploration.

    Returns
    -------
    np.ndarray
        The knowledge gradient values for each point in the input.
    """
    # Calculate the improvement (mean - best_observed - kappa)
    improvement = mean - best_observed - kappa

    # Calculate the Z value for standardization
    Z = improvement / (standard_deviation + 1e-9)  # Adding epsilon to avoid division by zero

    # Calculate the Expected Improvement for the next step
    ei = improvement * norm.cdf(Z) + standard_deviation * norm.pdf(Z)

    # Compute the Knowledge Gradient
    kg = ei / (standard_deviation + 1e-9)

    return kg

def BayesianExpectedLoss(mean, standard_deviation, best_observed=1.0, kappa=0.01):
    """
    Compute the Bayesian Expected Loss (BEL) acquisition function value for a given set of parameters.

    The BEL method calculates the expected loss associated with selecting a point that is not the true 
    optimum, considering the uncertainty of the predictions.

    Parameters
    ----------
    mean : np.ndarray
        The predicted mean values of the objective function.
    standard_deviation : np.ndarray
        The predicted standard deviation (uncertainty) of the prediction.
    best_observed : float, optional
        The current best observed value of the objective function.
    kappa : float, optional
        Exploration parameter, a small positive value to encourage exploration.

    Returns
    -------
    np.ndarray
        The BEL values for each point in the input.
    """
    # Calculate the improvement (mean - best_observed - kappa)
    improvement = mean - best_observed - kappa
    
    # Calculate the Z value for standardization
    Z = improvement / (standard_deviation + 1e-9)  # Adding epsilon to avoid division by zero
    
    # Calculate the loss: expected loss is proportional to the distance from the best observed
    loss = norm.pdf(Z) * standard_deviation + (Z * norm.cdf(Z)) * standard_deviation
    
    # Calculate the Bayesian Expected Loss
    bel = loss
    
    return bel

# ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                            
# ==============----------------- -- -- - Load/Save Object - -- -- -------------------================ #    

def SaveOptimiser(object, file_path):
    """
    Save the Bayesian Optimization object to a file using pickle.

    This function serialises the given Bayesian Optimization object and saves it to 
    a specified file path. This allows for the persistence of the optimiser's state,
    enabling it to be loaded and used later.

    Parameters
    ----------
    object : BO
        The Bayesian Optimization object to be saved.
    file_path : str
        The file path where the object should be saved.
    """
    # Open the specified file in write-binary mode
    with open(file_path, 'wb') as file:
        # Serialise the object using pickle and write it to the file
        pickle.dump(object, file)

def LoadOptimiser(file_path):
    """
    Load a Bayesian Optimization object from a file using pickle.

    This function deserialises a Bayesian Optimization object from a specified file path,
    allowing for the continuation of a previously saved optimization process.

    Parameters
    ----------
    file_path : str
        The file path from where the object should be loaded.

    Returns
    -------
    BO
        The loaded Bayesian Optimization object.
    """
    # Open the specified file in read-binary mode
    with open(file_path, 'rb') as file:
        # Deserialise the object using pickle and return it
        object = pickle.load(file)

    return object




# ==============----------------- -- -- -- - - - - - - -- -- -- -- -------------------================ #
                                            
# ==============----------------- -- -- - - - Plotting - - - -- -- -------------------================ #   

def SausagePlot(object, highlight_recent=0, resolution=1000, normalised=False):
    """
    Generate a "sausage plot" to visualise the mean and variance of a one-dimensional function.

    Parameters
    ----------
    object : BO
        The Bayesian Optimization object to be visualised.
    highlight_recent : int, optional
        The number of most recent points to highlight on the plot.
    resolution : int, optional
        The number of points used to sample the function for plotting.
    """
    plt.figure(figsize=(12, 6))
    
    if len(object.bounds) == 1:
        # Generate sample points for the plot
        sample_points = np.linspace(object.bounds[0][0], object.bounds[0][1], resolution, endpoint=True).reshape(resolution,1)
        # sample_points = np.linspace(0, 1, resolution, endpoint=True).reshape(resolution,1)

        # Predict mean and variance at the sample points
        mean, variance = object.PredictMeanVariance(sample_points)

        y_min = np.min(object.y_data)
        y_max = np.max(object.y_data)
        actual_mean = y_min + (y_max - y_min) * (mean + 1) / 2

        variance_scaling_factor = ((y_max - y_min) / 2) ** 2
        actual_variance = variance * variance_scaling_factor

        plt.figure(figsize=(12, 7))

        if normalised==False:
            # Plot the mean and the confidence interval
            plt.plot(sample_points, actual_mean, label='mean', color='k')
            plt.fill_between(sample_points[:,0], actual_mean[:,0] - 1.96 * np.sqrt(actual_variance[:,0]), actual_mean[:,0] + 1.96 * np.sqrt(actual_variance[:,0]), color = 'k', alpha=0.25, label = '95% confidence interval')

            # Scatter plot of X_data and y_data
            plt.scatter(object.X_data, object.y_data, s=50, color='k')
            
            if highlight_recent != 0:
                # Highlight the most recent points
                plt.scatter(object.X_data[-highlight_recent:], object.y_data[-highlight_recent:], s=100, color='red', label=f'most recent {highlight_recent} points')

            plt.title("Mean/Varance Plot")
            plt.legend()
            plt.show()

        if normalised==True:
            # normalise the y_data for consistency in Gaussian Process calculations
            if np.max(object.y_data) - np.min(object.y_data) != 0.0:
                normalised_y_data = 2 * (object.y_data - np.min(object.y_data)) / (np.max(object.y_data) - np.min(object.y_data)) - 1
            else:
                normalised_y_data = object.y_data

            if object.minimise is True:
                normalised_y_data = -normalised_y_data
            
            # Plot the mean and the confidence interval
            plt.plot(sample_points, mean, label='mean', color='k')
            plt.fill_between(sample_points[:,0], mean[:,0] - 1.96 * np.sqrt(variance[:,0]), mean[:,0] + 1.96 * np.sqrt(variance[:,0]), color = 'k', alpha=0.25, label = '95% confidence interval')

            # Scatter plot of X_data and y_data
            plt.scatter(object.X_data, normalised_y_data, s=50, color='k')
            
            if highlight_recent != 0:
                # Highlight the most recent points
                plt.scatter(object.X_data[-highlight_recent:], normalised_y_data[-highlight_recent:], s=100, color='red', label=f'most recent {highlight_recent} points')

            plt.title("Mean/Varance Plot")
            plt.legend()
            plt.show()

    else:
        print('Can only produce sausage plots of one dimensional functions.')

def KappaAcquisitionFunctionPlot(object, number_kappas, number_candidate_points, max_kappa, min_kappa, resolution=1000):
    """
    visualise the acquisition function for a range of kappa values.

    This function calculates and plots the acquisition function for a specified number of 
    kappa values. It generates candidate X points, calculates the corresponding Y values, 
    and identifies the maximum Y value for each kappa. The results are plotted to help 
    visualise the acquisition function and its dependency on the kappa value.

    Parameters
    ----------
    object : BO
        The Bayesian Optimization object that contains methods to calculate kappa, mean, and variance.
    number_kappas : int
        The number of kappa values to test.
    number_candidate_points : int
        The number of candidate X points to sample.
    max_kappa : float
        The maximum kappa value.
    min_kappa : float
        The minimum kappa value.
    resolution : int, optional
        The number of points used to sample the acquisition function.
    """
    plt.figure(figsize=(12, 6))

    # Function requires a one-dimensional optimization problem
    if len(object.bounds) == 1:
        kappas = np.empty([number_kappas, 1])  # initialise an empty array to store kappa values
        
        # Calculate kappa values for the specified range
        for i in range(number_kappas):
            kappas[i] = object.CalculateKappa(number_kappas, i, max_kappa, min_kappa)

        # Generate evenly spaced sample points between the bounds
        sample_points = np.concatenate([np.linspace(lower_bound, upper_bound, resolution, endpoint=True) for (lower_bound, upper_bound) in object.bounds]).reshape(resolution,1)
        
        # Predict the mean and variance for the sample points
        mean, variance = object.PredictMeanVariance(sample_points)

        x_values = []

        # Iterate through each kappa value to calculate the acquisition function
        for i in range(number_kappas):
            candidate_X = np.empty([number_candidate_points, len(object.bounds)])  # initialise array to store candidate X points
            candidate_y = np.empty([number_candidate_points, 1])  # initialise array to store corresponding y values
            
            # Generate candidate points and calculate their acquisition values
            for j in range(number_candidate_points):
                random_index = np.random.randint(0, resolution)  # Randomly select an index
                candidate_X[j] = sample_points[random_index]  # Store the corresponding X value
                candidate_y[j] = object.AcquisitionFunction(mean[random_index], np.sqrt(variance[random_index]), kappas[i])  # Calculate and store the acquisition value
                                               
            # Calculate the acquisition function for the entire sample space
            sample_y = object.AcquisitionFunction(mean, np.sqrt(variance), kappas[i])

            # Plot the acquisition function curve
            plt.plot(sample_points, sample_y, label=f'Kappa={round(kappas[i][0],2)}', color=f'C{i}')

            # Choose the x value which corresponds to the largest candidate y
            max_index = np.argmax(candidate_y)

            x_values.append(candidate_X[max_index])

            # Plot the point with the maximum acquisition value
            plt.scatter(candidate_X[max_index], candidate_y[max_index], s=50, color=f'C{i}')
            
            # Plot the other acquisition values
            plt.scatter(candidate_X, candidate_y, s=10,  color=f'C{i}')

        # Label the axes
        plt.xlabel('X')
        plt.ylabel('Acquisition Function Value')
        plt.title("Acquisition Function")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

        return np.array(x_values)
    else:
        print('Function requires a one dimensional optimization problem.')

def PlotData(object, legend=False):
    """
    Plot the objective function values against simulation numbers.

    Parameters
    ----------
    object : BO
        The Bayesian Optimization object containing data to be plotted.
    """
    plt.figure(figsize=(12, 6))
    iteration_numbers = np.unique(object.iterations_array)
    next_batch_simulation_number = 0

    for i in iteration_numbers:
        indices = np.where(object.iterations_array == i)[0]
        simulation_numbers = np.arange(next_batch_simulation_number, next_batch_simulation_number + len(indices))
        plt.scatter(simulation_numbers, object.y_data[indices], s=20, label=f'Iteration {int(i)}')
        next_batch_simulation_number = np.max(simulation_numbers) + 1

    plt.title('Objective function value against simulation number')
    plt.xlabel('Simulation number')
    plt.ylabel('Objective function value')
    if legend:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def PlotParameterEvolution(object):
    """
    Plot the evolution of optimization parameters over simulation runs.

    This function generates a scatter plot for each optimization parameter
    to visualise how each parameter value evolves over the course of simulations.

    Parameters
    ----------
    object : BO
        The Bayesian Optimization object containing data to be plotted.
    """
    COLUMN_NUMBER = 3  # Define the number of columns in the plot grid

    # Create a figure and a set of subplots with a dynamic number of rows
    fig, ax = plt.subplots(nrows=int(np.ceil(len(object.X_data[0]) / COLUMN_NUMBER)), ncols=COLUMN_NUMBER, figsize=(10, 10))

    if np.ndim(ax) == 1:
        # If ax is one-dimensional, iterate over each parameter
        for k in range(len(object.X_data[0])):
            ax[k].scatter(range(len(object.X_data)), object.X_data[:, k])
            ax[k].set_ylabel('X%i' %k)
            ax[k].set_xlabel('Simulation #')
            ax[k].title.set_text('X%i' %k)
    else:
        # If ax is two-dimensional, iterate over each parameter
        for k in range(len(object.X_data[0])):
            ax[k//COLUMN_NUMBER, k%COLUMN_NUMBER].scatter(range(len(object.X_data)), object.X_data[:, k])
            ax[k//COLUMN_NUMBER, k%COLUMN_NUMBER].set_ylabel('X%i' %k)
            ax[k//COLUMN_NUMBER, k%COLUMN_NUMBER].set_xlabel('Simulation #')
            ax[k//COLUMN_NUMBER, k%COLUMN_NUMBER].title.set_text('X%i' %k)

    # Adjust layout to prevent subplot labels from overlapping
    plt.tight_layout()

def PlotParameterCorrelation(object):
    """
    Plot the correlation between each optimization parameter and the objective value.

    This function creates scatter plots to visualise the relationship between the 
    first optimization parameter (X0) and every other parameter, colored by the 
    corresponding objective values.

    Parameters
    ----------
    object : BO
        The Bayesian Optimization object containing data to be plotted.
    """
    COLUMN_NUMBER = 3  # Define the number of columns in the plot grid

    # Create a figure and a set of subplots with a dynamic number of rows
    fig, ax = plt.subplots(nrows=int(np.ceil(len(object.X_data[0]) / COLUMN_NUMBER)), ncols=COLUMN_NUMBER, figsize=(15, 15))

    if np.ndim(ax) == 1:
        # If ax is one-dimensional, iterate over each parameter
        for k in range(len(object.X_data[0])):
            scatter = ax[k].scatter(object.X_data[:, 0], object.X_data[:, k], c=object.y_data, cmap='viridis')
            ax[k].set_ylabel('X%i' %k)
            ax[k].set_xlabel('X0')
            ax[k].title.set_text('X0 / X%i' %k)  
    else: 
        # If ax is two-dimensional, iterate over each parameter
        for k in range(len(object.X_data[0])):
            scatter = ax[k//COLUMN_NUMBER, k%COLUMN_NUMBER].scatter(object.X_data[:, 0], object.X_data[:, k], c=object.y_data, cmap='viridis')
            ax[k//COLUMN_NUMBER, k%COLUMN_NUMBER].set_ylabel('X%i' %k)
            ax[k//COLUMN_NUMBER, k%COLUMN_NUMBER].set_xlabel('X0')
            ax[k//COLUMN_NUMBER, k%COLUMN_NUMBER].title.set_text('X0 / X%i' %k)

    # Add a colorbar to the figure to represent the objective values
    fig.colorbar(scatter, ax=ax.ravel().tolist(), label='Y value')

    plt.show()