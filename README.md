# Bayesian Optimisation

A Python package for global optimisation using Gaussian Processes, designed to optimise expensive black-box functions with minimal evaluations.


## How Bayesian Optimisation Works


Bayesian optimisation works by building a probabilistic model of the objective function to make decisions on where to evaluate the function next. It aims to maximise (or minimise) the objective function in as few evaluations as possible. Here is a more detailed breakdown of the process:

1. **Constructing the Prior Model:** The optimisation process begins by defining a Gaussian Process (GP) as a model for the objective function. The GP uses a kernel function, with an associated hyper-parameters, and a mean to provide a prior distribution over possible functions that could describe the objective function.

2. **Condition the Model on Observations:** The GP model is updated with the current datapoints, transforming the prior into a posterior distribution. The posterior distribution represents the updated set of possible functions that could describe the objective function, given the observed datapoints. The mean and variance of this set of functions are then used by the acquisition function to select the next sample points.

3. **Selecting the Next Sample Points:** To determine where to evaluate the objective function next, Bayesian optimization uses an acquisition function. The acquisition function uses the posterior GP's mean and variance predictions, as well as a parameter 'kappa' which balances the exploration (sampling in areas with high uncertainty to gain more information) and exploitation (sampling in areas predicted to have high function values to optimise the function further), to assign a value to any point in parameter space. The acquisition function is evaluated at a set of random points and the one with the highest value is chosen as the point to sample next, guiding the search towards the optimal solution.

4. **Evaluate the Objective Function:** The Objective function is evaluated at the point selected by the acquistion function and the data are updated with both the point and its corresponding value. 

5. **Iterative Optimisation Process:** The optimisation process works by repeating steps 1-4.


## Key Features

- **Multiple Kernel Functions:** Radial Basis Function (RBF) and Matern kernels are included in this package, while there is also the ability to add your own kernel function.

- **Dynamic Bounds Reduction:** There is the option to automatically narrow the parameter space based on the observed data. As the optimisation progresses, the algorithm zooms in on regions of the parameter space where the best result has been observed, improving the resolution around the region and can significantly reduce the number of evaluations needed, particularly in high-dimensional problems.

- **Batching and Sub-Batching:** Batching enables the evaluation of multiple points simulateously, which is particularly useful in paraellel computing environments, where multiple evaluations of the objective function can be performed concurrently. Sub-batching further refines this process by dividing the batch into smaller groups. For each sub-batch, points are determined by maximising the acquisition function; the GP model is then updated with the selected points but **not** the corresponding value of the objective function. This approach helsp the GP model more effectively guide the selection of points in subsequent sub-batchesin without needing to evaulate the costly objective function for each sub-batch. The goal is to achieve a better distribution of points within each batch, optimising the overall serach efficienct without signifciantly increasing the optimisation time.

- **Diverse Acquisition Functions:** The package includes a range of acquisition functions: Upper Confidence Bound (UCB), Expected Improvement (EI), Probability Improvement (PI), Knowledge Gradient (KG) and Bayesian Expected Loss (BEL). Each of these functions allow users balance the competing objectives of exploitation and exploration using the parameter 'kappa'.

- **Logging and Output:** Included are comprehensive logging capabilities to track the optimisation process in detail. Results can be ouputted to CSV files, making it easy to store data and analyse the optimisation progress.

- **Saving and Loading Data:** The optimiser can be saved at any point in the process and later reloaded. In addition, the optimiser supports the import of data from CSV files, allowing users to initialise the oprimiser with existing data or incorporate results from previous experiments. This can significantly reduce the number of evaluations needed of the objective function by starting from a well-informed state.

- **Visualisation Tools:** A variety of plotting functions are built in to visualise the optimisation process, such as predictions made by the GP model, acquisition function landscapes and parameter evolution.

## Installation

To install this package, you can use pip:

pip install git+https://github.com/Jor-Lee/Bayesian

## Example Notebooks

You will find a quick example to get you started in the 'Basic_example' python notebook.

'1D_example' contains an example of how you the package works to optimise a test function using batches.

'2D_example' shows how to use the package in higher dimensions.

'Bounds_reduction_example' demonstrates in detail how to optimise with bounds' reduction.

'Sub_batch_example' explains what sub-batching does and how to use it. 
