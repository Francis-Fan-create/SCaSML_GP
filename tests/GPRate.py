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

    def test(self, save_path, sample_sizes=range(100, 1001, 100)):
        '''
        Compares solvers on different distances on the sphere.

        Parameters:
        save_path (str): The path to save the results.
        sample_sizes (list): The number of samples for testing, should be a list to plot the convergence rate.

        '''
        #initialize the profiler
        profiler = cProfile.Profile()
        profiler.enable()      
        # create the save path if it does not exist
        class_name = self.__class__.__name__
        new_path = f"{save_path}/{class_name}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        save_path = new_path
        # Set the approximation parameters
        eq = self.equation
        eq_name = eq.__class__.__name__
        eq_dim=eq.n_input-1
        geom = eq.geometry()
        _,data_boundary=eq.generate_data(1,20)
        random_methods = ["LHS"]
        for random_method in random_methods:
            errors_list = []
            for index,sample_size in enumerate(sample_sizes):
                errors = np.zeros(sample_size)  # errors: ndarray, shape: (sample_size,), dtype: float

                # Compute the errors
                xt_values = geom.random_points(sample_size,random=random_method)  # xt_values: ndarray, shape: (sample_size, n_input), dtype: float
                exact_sol = eq.exact_solution(xt_values)  # exact_sol: ndarray, shape: (n_samples,), dtype: float

                # Measure the evaluation_number for solver1
                if index == 0:
                    sol1 = self.solver1.GPsolver(xt_values, data_boundary, GN_step=4)  # sol1: ndarray, shape: (num_domain,), dtype: float
                else:
                    sol1 = self.solver1.predict(xt_values)  # sol1: ndarray, shape: (num_domain,), dtype: float

                # Compute the errors
                errors=np.abs(sol1 - exact_sol)

                # Compute the mean errors
                errors_list.append(np.mean(errors))
            
            epsilon = 1e-10
            # Convert lists to arrays
            sample_sizes_array = np.array(sample_sizes)  # sample_sizes_array: ndarray, shape: (len(sample_sizes),), dtype: int
            errors_array = np.array(errors_list)  # errors_array: ndarray, shape: (len(sample_sizes),), dtype: float

            
            
            # Plot the convergence rate for GP
            plt.figure()
            plt.plot(np.log10(sample_sizes_array + epsilon), np.log10(np.array(errors_array) + epsilon), label='GP')
            slope_1_2 = -1/2 * (np.log10(sample_sizes_array+epsilon)-np.log10(sample_sizes_array[0]+epsilon)) + np.log10(errors_array[0] + epsilon)
            slope_1_4 = -1/4 * (np.log10(sample_sizes_array+epsilon)-np.log10(sample_sizes_array[0]+epsilon)) + np.log10(errors_array[0] + epsilon)
            plt.plot(np.log10(sample_sizes_array + epsilon), slope_1_2, label='slope=-1/2')
            plt.plot(np.log10(sample_sizes_array + epsilon), slope_1_4, label='slope=-1/4')
            plt.scatter(np.log10(sample_sizes_array + epsilon), np.log10(np.array(errors_array) + epsilon), marker='x')
            plt.scatter(np.log10(sample_sizes_array + epsilon), slope_1_2, marker='x')
            plt.scatter(np.log10(sample_sizes_array + epsilon), slope_1_4, marker='x')
            plt.title(f'GP - Sample Rate {random_method}')
            plt.xlabel('log10(sample_size)')
            plt.ylabel('log10(error)')
            plt.legend()
            plt.savefig(f'{save_path}/GP_sample_rate_{random_method}.png')
            wandb.log({f'GP_sample_rate_{random_method}': plt})
            
        
        return 0