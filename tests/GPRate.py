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
        Compares solvers on different distances on the sphere.

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
        eq_name = eq.__class__.__name__
        eq_dim = eq.n_input - 1
        geom = eq.geometry()

        # Generate boundary data (assuming needed for GPsolver)
        _, data_boundary = eq.generate_data(1, 50)

        # Define the sampling methods (e.g., Latin Hypercube Sampling)
        random_methods = ["LHS"]

        for random_method in random_methods:
            errors_list = []

            # Step 1: Generate a fixed test dataset
            test_sample_size = 1000  # Fixed size for test data
            test_xt_values = geom.random_points(test_sample_size, random=random_method)  # Fixed test inputs
            test_exact_sol = eq.exact_solution(test_xt_values)  # Fixed test outputs

            for index, sample_size in enumerate(sample_sizes):
                # Step 2: Generate training data of the current sample size
                training_sample_size = sample_size
                training_xt_values = geom.random_points(training_sample_size, random=random_method)  # Training inputs
                training_exact_sol = eq.exact_solution(training_xt_values)  # Training outputs

                # Step 3: Train or update the GP model
                self.solver1.GPsolver(training_xt_values, data_boundary)
                # Step 4: Predict on the fixed test dataset
                sol1 = self.solver1.predict(test_xt_values)  # Predictions on test data

                # Step 5: Compute the errors between predictions and exact solutions
                errors = np.abs(sol1 - test_exact_sol)

                # Step 6: Compute the mean absolute error and store it
                mean_error = np.mean(errors)
                errors_list.append(mean_error)

                print(f"Sample Size: {sample_size}, Mean Absolute Error: {mean_error}")

            # Convert lists to numpy arrays for plotting
            sample_sizes_array = np.array(sample_sizes)  # Shape: (len(sample_sizes),)
            errors_array = np.array(errors_list)        # Shape: (len(sample_sizes),)
            epsilon = 1e-10  # To avoid log(0)

            # Step 7: Plot the convergence rate for GP
            plt.figure(figsize=(8, 6))

            # Plot the original data
            plt.plot(sample_sizes_array + epsilon, errors_array + epsilon, marker='x', linestyle='-', label='GP')

            # Define the reference point for calculating slopes
            x0 = sample_sizes_array[0] + epsilon
            y0 = errors_array[0] + epsilon

            # Calculate slope lines based on theoretical convergence rates
            # For slope = -1/2: error = C * (sample_size)^(-1/2)
            C1_2 = y0 * (x0 ** 0.5)
            slope_1_2 = C1_2 * (sample_sizes_array + epsilon) ** (-0.5)

            # For slope = -1/4: error = C * (sample_size)^(-1/4)
            C1_4 = y0 * (x0 ** 0.25)
            slope_1_4 = C1_4 * (sample_sizes_array + epsilon) ** (-0.25)

            # Plot the slope reference lines
            plt.plot(sample_sizes_array + epsilon, slope_1_2, linestyle='--', label='Slope = -1/2')
            plt.plot(sample_sizes_array + epsilon, slope_1_4, linestyle='--', label='Slope = -1/4')

            # Scatter markers for data points and slope lines
            plt.scatter(sample_sizes_array + epsilon, errors_array + epsilon, marker='x', color='blue')
            plt.scatter(sample_sizes_array + epsilon, slope_1_2, marker='o', color='orange')
            plt.scatter(sample_sizes_array + epsilon, slope_1_4, marker='^', color='green')

            # Set plot titles and labels
            plt.title(f'GP Convergence Rate - Sampling Method: {random_method}')
            plt.xlabel('Training Sample Size')
            plt.ylabel('Mean Absolute Error on Test Set')

            # Apply logarithmic scales to both axes
            plt.xscale('log')
            plt.yscale('log')

            # Add legend to distinguish between plots
            plt.legend()
            plt.grid(True, which="both", ls="--", linewidth=0.5)

            # Save the plot to the specified path
            plot_filename = f'GP_convergence_rate_{random_method}.png'
            plt.savefig(os.path.join(save_path, plot_filename))
            plt.close()

            # Log the plot using Weights & Biases (wandb) if required
            # Initialize wandb run if not already done
            # wandb.init(project='gprate', name='convergence_rate')
            # wandb.log({f'GP_convergence_rate_{random_method}': wandb.Image(os.path.join(save_path, plot_filename))})
            # wandb.finish()

            print(f"Convergence plot saved to {os.path.join(save_path, plot_filename)}")

        # Disable the profiler and print stats
        profiler.disable()
        profiler.print_stats(sort='cumtime')

        return 0