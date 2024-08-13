### think about changing np to cp / running matrix multiplication on GPU

import os
import time
import sys
import pandas as pd
import heapq
import numpy as np
import logging

class BO:
    def __init__(self, log_path, KernelFunction, AcquisitionFunction, bounds, n_samples, length_scale, iterations, batch_size, max_kappa, min_kappa, output_directory, random_seed, iterations_between_reducing_bounds, first_reduce_bounds, reduce_bounds_factor):
        """
        Initialize the Bayesian Optimization (BO) class with various parameters.

        Parameters:
        - log_path (str): Path to the log file.
        - KernelFunction (function): Function representing the kernel used in the Gaussian Process.
        - AcquisitionFunction (function): Function to calculate the acquisition value.
        - bounds (list): List of tuples representing the bounds for the parameters.
        - n_samples (int): Number of samples to take.
        - length_scale (float): The length scale for the kernel.
        - iterations (int): Number of iterations to run the optimization.
        - batch_size (int): Number of samples per batch.
        - max_kappa (float): Maximum kappa value for exploration-exploitation balance.
        - min_kappa (float): Minimum kappa value.
        - output_directory (str): Directory to save outputs.
        - random_seed (int): Seed for random number generation.
        - iterations_between_reducing_bounds (int): Number of iterations before reducing bounds.
        - first_reduce_bounds (int): Number of initial iterations before starting to reduce bounds.
        - reduce_bounds_factor (float): Factor by which to reduce bounds.
        """
        self.log_path = log_path
        self.Kernel = KernelFunction
        self.AcquisitionFunction = AcquisitionFunction
        self.bounds = bounds
        self.n_samples = n_samples
        self.length_scale = length_scale
        self.iterations = iterations
        self.batch_size = batch_size
        self.max_kappa = max_kappa
        self.min_kappa = min_kappa
        self.output_directory = output_directory
        self.random_seed = random_seed
        self.iterations_between_reducing_bounds = iterations_between_reducing_bounds
        self.first_reduce_bounds = first_reduce_bounds
        self.reduce_bounds_factor = reduce_bounds_factor
        self.random_counter = 0
        self.X_data = np.array([])
        self.y_data = np.array([])
        self.stuck_in_peak_counter = 0
        self.bounds_reduction_counter = 0
        self.current_best_value = 0

    def CreateLogger(self):
        """
        Create a logger for the optimization process.

        This function checks if a log file already exists. If it does, the program exits
        to prevent overwriting. If not, a new logger is created and configured.

        Raises:
        - SystemExit: If the log file already exists.
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

        self.logger.info('The log has been created')

    def GetRandomXBatch(self):
        """
        Generate a batch of random X values within the specified bounds.

        This method generates random points within the bounds for each batch of 
        simulations, ensuring that each point is unique within the batch.

        Returns:
        - np.ndarray: A 2D array of randomly generated X values for the batch.
        """

        optimiser_start_time = time.time()  # Record start time for optimization

        self.logger.info(f'Getting X values for the random iteration')
        self.logger.info('')

        raw_X = []  # Initialize the list to store the batch of X values

        # Set the random seed for reproducibility
        np.random.seed(self.random_seed)
        
        for i in range(self.batch_size):
            # Generate a random point within the specified bounds
            next_point = np.array([np.random.uniform(low, high) for (low, high) in self.bounds])

            # Ensure the point is unique within the batch
            counter = 0
            while list(next_point) in raw_X:
                self.logger.info(f'Simulation {i} produced a point that already exists...recalculating X values. Attempt {counter+1}')
                next_point = np.array([np.random.uniform(low, high) for (low, high) in self.bounds])
                counter += 1

            # Append the next point to the array of all points for this iteration
            raw_X.append(list(next_point))

        raw_X = np.array(raw_X)  # Convert the list of X values to a numpy array

        optimiser_end_time = time.time()  # Record end time for optimization
        self.logger.info(f'The time taken to get all X values for the random iteration was {(optimiser_end_time-optimiser_start_time)/60} minutes.')
        self.logger.info('')

        return raw_X

    def InverseKernel(self, iteration_number):
        """
        Calculate the inverse of the kernel matrix (K).

        This method adds a small jitter term to the kernel matrix for numerical stability,
        computes the inverse, and returns it.

        Parameters:
        - iteration_number (int): The current iteration number, used for logging.

        Returns:
        - np.ndarray: The inverse of the kernel matrix (K_inv).
        """
        jitter = 1e-7  # Small jitter value for numerical stability

        # Compute the kernel matrix with the jitter term added
        self.K = self.Kernel(self.X_data, self.X_data, self.length_scale) + jitter * np.eye(len(self.X_data))

        # Calculate the inverse of the kernel matrix
        inverting_start_time = time.time()  # Record start time for inversion
        self.K_inv = np.linalg.inv(self.K)
        inverting_end_time = time.time()  # Record end time for inversion
        
        # Log the time taken to invert the kernel matrix for the current iteration
        self.logger.info(f'It took {inverting_end_time-inverting_start_time} to invert the kernel for iteration {iteration_number}')

        return self.K_inv

    def PredictMeanStandardDeviation(self):
        """
        Predict the mean and standard deviation of the objective function.

        This method uses the Gaussian Process (GP) to predict the mean and variance of
        the objective function at random points within the specified bounds.
        """
        # Generate random samples within the bounds
        self.x_samples = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_samples, self.bounds.shape[0]))
        self.x_samples = np.array(self.x_samples)
        
        # Calculate the covariance vector between the training points and the new point
        self.k_star = self.Kernel(self.X_data, self.x_samples, self.length_scale)

        # Normalize the observed data
        normalized_y_data = self.y_data / np.max(self.y_data)

        # Predict the mean of the new point
        self.mean = self.k_star.T.dot(self.K_inv).dot(normalized_y_data)  

        # Add a small jitter term to k_star_star for numerical stability
        jitter = 1e-7
        
        # Predict the variance of the new point
        self.k_star_star = self.Kernel(self.x_samples, self.x_samples, self.length_scale) + jitter
        
        # Compute the full covariance matrix of the prediction
        self.full_cov = self.k_star_star - self.k_star.T.dot(self.K_inv).dot(self.k_star)

        # Extract the diagonal elements to get the variances for each new point
        self.var = np.diag(self.full_cov)

        # Compute the standard deviation (square root of variance)
        self.standard_deviation = np.sqrt(self.var)
    
    def CalculateKappa(self,current_simulation_number):
        """
        Compute the UCB parameter kappa for the current batch.

        Kappa determines the exploration-exploitation trade-off in Bayesian Optimization. 
        It is dynamically calculated based on the batch size and the current simulation number.

        Parameters:
        - current_simulation_number (int): The index of the current simulation within the batch.

        Returns:
        - float: The computed kappa value.
        """
        try:
            # Check if batch_size is equal to 1
            if self.batch_size == 1:
                # Compute kappa as the average of max_kappa and min_kappa
                self.kappa = (self.max_kappa - self.min_kappa) / 2
            else:
                # Calculate the exponential factor 'b'
                b = 1/(self.batch_size-1) * np.log(self.max_kappa / self.min_kappa)

                # Calculate kappa using an exponential function
                self.kappa = self.min_kappa * np.exp(b*current_simulation_number)

        # Handle any exceptions that occur during the try block
        except Exception as e:
            self.logger.info(f"An error occurred: {e}")   

        return self.kappa

    def GetNextX(self, current_simulation_number):
        """
        Get the next set of input parameters for the objective function.

        This method computes the next best set of input parameters by optimizing the
        acquisition function based on the predicted mean, standard deviation, and kappa value.

        Parameters:
        - current_simulation_number (int): The index of the current simulation within the batch.

        Returns:
        - np.ndarray: The next set of input parameters (X).
        """
        # Compute and invert the kernel matrix for the current simulation
        self.InverseKernel(iteration_number=0)

        # Calculate kappa for the current simulation
        self.kappa = self.CalculateKappa(current_simulation_number)

        # Predict the mean and standard deviation of the function at new points
        self.PredictMeanStandardDeviation()

        # Draw samples from the posterior using the acquisition function
        samples = self.AcquisitionFunction(self.mean, self.standard_deviation, self.kappa)

        # Identify the index of the maximum sample
        max_index = np.argmax(samples)

        # Select the next X value corresponding to the maximum sample
        next_X = self.x_samples[max_index]

        return next_X

    def UniquenessCheck(self, X, raw_X, iteration_number, simulation_number):
        """
        Ensure the generated point is unique within the batch.

        This method checks if the generated point already exists within the current batch. 
        If it does, a new point is generated until a unique one is found or a limit is reached. 
        If the limit is reached, a random point is generated.

        Parameters:
        - X (1D array): The generated point to be checked for uniqueness.
        - raw_X (2D array): The array of points already generated in the batch.
        - iteration_number (int): The current iteration number.
        - simulation_number (int): The index of the simulation in the current batch.

        Returns:
        - np.ndarray: A unique point for the batch.
        """
        counter = 0
        next_point = X

        # Loop until a unique point is found or a limit is reached
        while list(X) in raw_X:
            self.logger.info(f'Iteration {iteration_number}, simulation {simulation_number} produced a point that already exists...recalculating X values. Attempt {counter+1}')
            next_point = self.GetNextX(simulation_number)
            counter += 1
            if counter > 9:
                self.random_counter += 1
                self.logger.info(f'Random X values being used. This has been done {self.random_counter} times so far')
                next_point = np.array([np.random.uniform(low, high) for (low, high) in self.bounds])
                break

        return next_point

    def GetNextXBatch(self, iteration_number):
        """
        Get the next batch of input parameters for the objective function.

        This method generates a batch of input parameters by iteratively optimizing 
        the acquisition function. It accounts for the exploration-exploitation trade-off 
        using kappa values and ensures uniqueness of the points.

        Parameters:
        - iteration_number (int): The current iteration number.

        Returns:
        - np.ndarray: The next batch of input parameters (X).
        """
        optimiser_start_time = time.time()  # Record start time for optimization

        self.logger.info(f'Getting X values for iteration {iteration_number}')
        self.logger.info('')

        raw_X = []  # Initialize the list to store the batch of X values

        # Compute and invert the kernel matrix for the current iteration
        self.InverseKernel(iteration_number)

        for i in range(self.batch_size):
            # Calculate kappa for the current simulation within the batch
            self.kappa = self.CalculateKappa(i)

            # Predict the mean and standard deviation for the new points
            self.PredictMeanStandardDeviation()

            # Draw samples from the posterior using the acquisition function
            samples = self.AcquisitionFunction(self.mean, self.standard_deviation, self.kappa)

            # Identify the index of the maximum sample
            max_index = np.argmax(samples)

            # Select the next point corresponding to the maximum sample
            next_point = self.x_samples[max_index]

            # Ensure the uniqueness of the point within the batch
            next_point = self.UniquenessCheck(next_point, raw_X, iteration_number, i)
            
            # Add the next unique point to the batch of X values after converting it to a list
            raw_X.append(list(next_point))
        
        raw_X = np.array(raw_X)  # Convert the list of X values to a numpy array

        optimiser_end_time = time.time()  # Record end time for optimization
        self.logger.info(f'The time taken to get all X values for iteration {iteration_number} was {(optimiser_end_time-optimiser_start_time)/60} minutes.')
        self.logger.info('')

        return raw_X

    def StuckInPeak(self):
        """
        Check if the optimizer has become stuck at a peak.

        This method compares the best value from the most recent iteration with the 
        current best value. If no improvement is observed, it increases a counter 
        that tracks how long the optimizer has been stuck in a peak.
        """
        # Define the range of indices for the most recent iteration
        relevant_indices = range(max(len(self.y_data) - self.batch_size, 0), len(self.y_data))
        
        # Find the largest index from the most recent iteration
        largest_index = heapq.nlargest(1, relevant_indices, key=self.y_data.__getitem__)
        
        # Retrieve the actual Y values for these indices
        largest_value = [self.y_data[i] for i in largest_index]

        # Check if the largest value found is greater than the current best value
        if largest_value[0] > self.current_best_value:
            self.stuck_in_peak_flag = 0  # Not stuck in a peak
            self.stuck_in_peak_counter = 0  # Reset the counter
            self.current_best_value = largest_value[0]  # Update the current best value
        else:
            self.stuck_in_peak_flag = 1  # Stuck in a peak
            self.stuck_in_peak_counter += 1  # Increment the counter
            self.logger.info('The Optimiser has become stuck at a peak')  # Log the event
    
    def FindBounds(self, number, range):
        """
        Find the upper and lower bounds for the domain for a given number within a specified range.

        This method calculates the bounds for a given number, ensuring that the bounds 
        remain within the interval [0, 1]. This is particularly useful when dealing with 
        normalized data or probabilities.

        Parameters:
        - number (float): The central value for which to find bounds.
        - range (float): The range around the central value.

        Returns:
        - list: A list containing the lower and upper bounds.
        """
        # Calculate the upper bound by adding half of the range to the number
        upper_bound = number + range/2

        # Calculate the lower bound by subtracting half of the range from the number
        lower_bound = number - range/2

        # Check if the lower bound is less than 0
        if lower_bound < 0:
            lower_bound = 0   # If the lower bound is less than 0, set it to 0
            upper_bound = range  # Adjust the upper bound to be the full range from 0

        # Check if the upper bound exceeds 1    
        elif upper_bound > 1:
            lower_bound = 1-range  # Adjust the lower bound to be 1 minus the range
            upper_bound = 1   # If the upper bound exceeds 1, set it to 1

        return [lower_bound,upper_bound]
    
    def ReduceBounds(self, iteration_number):
        """
        Reduce the search bounds if stuck in a peak.

        This method reduces the search bounds if the optimizer has been stuck in a peak 
        for a specified number of iterations. It proportionally reduces the bounds and 
        the length scale used in the kernel function.

        Parameters:
        - iteration_number (int): The current iteration number.
        """
        self.logger.info(f'Stuck in peak counter is: {self.stuck_in_peak_counter}')

        # Reduce the search bounds if stuck in a peak and past the first_reduce_bounds threshold
        if self.stuck_in_peak_counter >= self.iterations_between_reducing_bounds and iteration_number >= self.first_reduce_bounds:
            self.stuck_in_peak_counter = 0  # Reset the stuck_in_peak_counter
            self.bounds_reduction_counter += 1  # Increment the bounds_reduction_counter
            spread = self.reduce_bounds_factor ** self.bounds_reduction_counter  # Set the correct range of the bounds
            self.length_scale = self.length_scale * self.reduce_bounds_factor  # Reduce the length scale proportionally to the bounds
            
            # Calculate the new bounds for this dimension using the current best X value and the spread
            bounds = []
            for i in range(len(self.bounds)):
                bounds.append(self.FindBounds(self.X_data[self.BestData()[0][0]][i],spread))
            self.bounds = np.array(bounds)

            self.logger.info(f'New bounds are {self.bounds}')

    def BestData(self):
        """
        Find the best data points in terms of the maximum observed values.

        This method identifies the largest observed values from the optimization process,
        sorts them, and returns the sorted indices and values.

        Parameters:
        - y_data (list): The list of observed values.

        Returns:
        - tuple: A tuple containing the sorted indices and values.
        """
        # Set the number of indices to find
        number_indices = 1

        # Find the indices of the largest Y values
        largest_indices = heapq.nlargest(number_indices, range(len(self.y_data)), key=self.y_data.__getitem__)
        
        # Retrieve the Y values for these indices
        largest_values = [self.y_data[i] for i in largest_indices]

        # Sort the indices and values into value order
        sorted_indices_and_values = sorted(zip(largest_indices, largest_values), key=lambda x: x[1], reverse=True)

        # Unzip the sorted pairs back into separate lists of indices and values
        sorted_indices, sorted_values = zip(*sorted_indices_and_values)

        return(sorted_indices, sorted_values)
   
    def WriteOutputToCSV(self, raw_X, raw_y, iteration_number):
        """
        Write the simulation results to a CSV file.

        This method saves the results of the current iteration, including the input 
        parameters (X) and output results (Y), to a CSV file. If the file does not exist, 
        it creates a new one with headers. Otherwise, it appends the new data to the existing file.

        Parameters:
        - raw_X (2D array): The input parameters used in the current iteration.
        - raw_y (1D array): The output results corresponding to the input parameters.
        - iteration_number (int): The current iteration number, used to tag the data.
        """
        self.csv_file = '%s/Results.csv' %self.output_directory  # Define the path to the CSV file

        # Create arrays for iteration numbers and simulation numbers
        iteration_numbers = np.full(len(raw_X), iteration_number)
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
        if not os.path.isfile(self.csv_file):
            df.to_csv(self.csv_file, index=False)
        else:
            # Append new data to the existing CSV file
            df.to_csv(self.csv_file, mode='a', header=False, index=False)

        self.logger.info('csv file updated.')

    def UpdateData(self, raw_X, raw_y):
        """
        Update the internal data storage with new X and Y values.

        This method appends the new input parameters (X) and output results (Y) from 
        the current iteration to the existing data arrays stored in the class.

        Parameters:
        - raw_X (2D array): The new input parameters to append.
        - raw_y (1D array): The new output results to append.
        """
        if self.X_data.size == 0:
            # If X_data is empty, initialize it with raw_X
            self.X_data = raw_X
        else:
            # Otherwise, concatenate raw_X to the existing X_data array
            self.X_data = np.concatenate((self.X_data, raw_X), axis=0)

        if self.y_data.size == 0:
            # If y_data is empty, initialize it with raw_y
            self.y_data = raw_y.flatten()
        else:
            # Otherwise, append the flattened raw_y to the existing y_data array
            self.y_data = np.append(self.y_data, raw_y.flatten())

    def UpdateFromCSV(self, csv_file):
        """
        Update the internal data storage by reading values from a CSV file.

        This method reads a CSV file containing results from previous iterations 
        and updates the internal data storage (X_data and y_data) with the information.

        Parameters:
        - csv_file (str): The path to the CSV file containing previous results.
        """
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # Extract the Y data from the DataFrame
        self.y_data.extend(df['Result'].values)

        # Calculate the current number of iterations
        current_number_iterations = int(len(self.y_data) / self.batch_size)

        # Extract the X data from the DataFrame
        self.X_data.extend(np.zeros((current_number_iterations * self.batch_size, len(self.bounds))).tolist())
        for i in range(current_number_iterations * self.batch_size):
            for k in range(len(self.bounds)):
                self.X_data[current_number_iterations+i][k] = df[f'X{k}'][i]

    def CurrentInfoStatus(self):
        """
        Log the current status of the optimization process.

        This method logs the best value found so far, the corresponding X values, 
        the number of random X values used, and how many times the bounds have been reduced.
        """
        self.logger.info(f'Current best y value was {self.BestData()[1][0]}; the corresponding X values were {self.X_data[self.BestData()[0][0]]}')
        self.logger.info(f'Current number of random X values is {self.random_counter}')
        self.logger.info(f'The bounds have been reduced {self.bounds_reduction_counter} times')
        self.logger.info('')
        self.logger.info('')

    def TotalInfoStatus(self):
        """
        Log the final status of the optimization process.

        This method logs the final best value found, the corresponding X values, 
        the total number of random X values used, and the total number of times the bounds were reduced.
        """
        self.logger.info(f'Best y value was {self.BestData()[1][0]}; the corresponding X values were {self.X_data[self.BestData()[0][0]]}')
        self.logger.info(f'Total number of random X values was {self.random_counter}')
        self.logger.info(f'The bounds were reduced a total of {self.bounds_reduction_counter} times')