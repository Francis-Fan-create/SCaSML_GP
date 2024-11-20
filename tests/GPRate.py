import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import time
import sys
import os
import cProfile
import shutil
import copy

class GPRate(object):
    '''
    GP Rate test in high dimensions.

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A PyTorch model for the GP network.
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    '''
    def __init__(self, equation, solver1):
        '''
        Initializes the normal spheres with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The GP network solver.
        '''
        #save original stdout and stderr
        self.stdout=sys.stdout
        self.stderr=sys.stderr
        # Initialize the normal spheres
        self.equation = equation
        self.dim = equation.n_input - 1  # equation.n_input: int
        self.solver1 = solver1
        self.t0 = equation.t0  # equation.t0: float
        self.T = equation.T  # equation.T: float
def test(self, save_path, sample_sizes=range(200, 600, 100)):
    '''
    Compares solvers on different training sample sizes.

    Parameters:
    save_path (str): The path to save the results.
    sample_sizes (list): The number of training samples, should be a list to plot the convergence rate.
    '''
    # Initialize the profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Create the save path if it does not exist
    class_name = self.__class__.__name__
    new_path = f"{save_path}/{class_name}"
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    save_path = new_path

    # Set the approximation parameters
    eq = self.equation
    eq_dim = eq.n_input - 1
    geom = eq.geometry()

    # Define the sampling methods (e.g., Latin Hypercube Sampling)
    random_methods = ["LHS"]

    for random_method in random_methods:
        errors_list = []

        # Step 1: Generate a fixed test dataset
        test_sample_size = 100  # Fixed size for test data
        test_xt_values = geom.random_points(test_sample_size, random=random_method)  # Fixed test inputs
        test_exact_sol = eq.exact_solution(test_xt_values)  # Fixed test outputs

        for sample_size in sample_sizes:
            # Step 2: Generate training data of the current sample size
            training_xt_values = geom.random_points(sample_size, random=random_method)  # Training inputs
            _, data_boundary = eq.generate_data(1, sample_size // 5)  # Training outputs

            # Step 3: Train the GP model
            self.solver1.GPsolver(training_xt_values, data_boundary)

            # Step 4: Predict on the fixed test dataset
            sol1 = self.solver1.predict(test_xt_values)  # Predictions on test data

            # Step 5: Compute the relative absolute error
            errors = np.abs(sol1 - test_exact_sol)
            mean_error = np.mean(errors)
            mean_exact_sol = np.mean(np.abs(test_exact_sol) + 1e-6)
            rel_error = mean_error / mean_exact_sol
            errors_list.append(rel_error)

            print(f"Sample Size: {sample_size}, Mean Relative Absolute Error: {rel_error}")

        # Convert lists to numpy arrays for plotting
        sample_sizes_array = np.array(sample_sizes)
        errors_array = np.array(errors_list)
        epsilon = 1e-10  # To avoid log(0)

        # Step 7: Plot the convergence rate for GP
        plt.figure(figsize=(8, 6))

        # Plot the data on log-log scale
        plt.plot(sample_sizes_array, errors_array, marker='x', linestyle='-', label='GP')

        # Fit a line to the log-log data to find the slope
        log_sample_sizes = np.log10(sample_sizes_array + epsilon)
        log_errors = np.log10(errors_array + epsilon)
        slope, intercept = np.polyfit(log_sample_sizes, log_errors, 1)
        fitted_line = 10 ** (intercept + slope * log_sample_sizes)
        plt.plot(sample_sizes_array, fitted_line, linestyle='--', label=f'Fitted Line (Slope: {slope:.2f})')

        # Set plot titles and labels
        plt.title(f'GP Convergence Rate - Sampling Method: {random_method}')
        plt.xlabel('Training Sample Size')
        plt.ylabel('Mean Relative Absolute Error on Test Set')

        # Apply logarithmic scales to both axes
        plt.xscale('log')
        plt.yscale('log')

        # Add legend and grid
        plt.legend()
        plt.grid(True, which="both", ls="--", linewidth=0.5)

        # Save the plot to the specified path
        plot_filename = f'GP_convergence_rate_{random_method}.png'
        plt.savefig(os.path.join(save_path, plot_filename))
        plt.close()

        print(f"Convergence plot saved to {os.path.join(save_path, plot_filename)}")

    # Disable the profiler and print stats
    profiler.disable()
    profiler.print_stats(sort='cumtime')

    return 0