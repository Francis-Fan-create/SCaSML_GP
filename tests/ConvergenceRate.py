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

class ConvergenceRate(object):
    '''
    Convergence Rate test in high dimensions.

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A PyTorch model for the GP network.
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    '''
    def __init__(self, equation, solver1, solver2, solver3):
        '''
        Initializes the normal spheres with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The GP network solver.
        solver2 (object): The MLP solver object.
        solver3 (object): The ScaSML solver object.
        '''
        #save original stdout and stderr
        self.stdout=sys.stdout
        self.stderr=sys.stderr
        # Initialize the normal spheres
        self.equation = equation
        self.dim = equation.n_input - 1  # equation.n_input: int
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.t0 = equation.t0  # equation.t0: float
        self.T = equation.T  # equation.T: float

    def test(self, save_path, rhomax=2, n_samples=50):
        '''
        Compares solvers on different training iterations.
    
        Parameters:
        save_path (str): The path to save the results.
        rhomax (int): The fixed value of rho for approximation parameters.
        n_samples (int): The number of samples for testing (test set).
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
        directory = f'{save_path}/callbacks'
    
        # Delete former callbacks
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
    
        # Set the approximation parameters
        eq = self.equation
        eq_dim = eq.n_input - 1
        geom = eq.geometry()
    
        # Fix rho to rhomax
        rho_ = rhomax
        self.solver2.set_approx_parameters(rho_)
        self.solver3.set_approx_parameters(rho_)
    
        # Fix training sample sizes
        train_domain = 100
        train_boundary = 20
        # Define a range of training iterations
        GN_steps_list = range(200, 1100, 100)
        error1_list = []
        error_ratio3_list = []
        GN_steps_output_list = []
    
        # Generate test data (fixed)
        xt_values = geom.random_points(n_samples, random="LHS")
        exact_sol = eq.exact_solution(xt_values)
    
        # Generate training data once
        data_domain_train, _ = eq.generate_data(train_domain, 0)
        _, data_boundary = eq.generate_data(1, train_boundary)
    
        for GN_steps in GN_steps_list:
            print(f"Training solver1 with {GN_steps} iterations...")
            # Train solver1 with fixed training sample size and varying GN_steps
            self.solver1.GPsolver(data_domain_train, data_boundary, GN_steps=GN_steps)
        
            # Predict with solver1
            sol1 = self.solver1.predict(xt_values)
        
            # Solve with solver2 (baseline solver)
            sol2 = self.solver2.u_solve(rho_, rho_, xt_values)
        
            # Solve with solver3 using the trained solver1
            sol3 = self.solver3.u_solve(rho_, rho_, xt_values)
        
            # Compute errors
            errors1 = (sol1 - exact_sol) ** 2
            errors2 = (sol2 - exact_sol) ** 2
            errors3 = (sol3 - exact_sol) ** 2
        
            # Compute error ratios
            error_value1 = np.mean(errors1)
            error_ratio3 = np.mean(errors3) / (np.mean(errors2) + 1e-6)
        
            error1_list.append(error_value1)
            error_ratio3_list.append(error_ratio3)
            GN_steps_output_list.append(GN_steps)
        
        # Plot error ratios
        plt.figure()
        plt.plot(GN_steps_output_list, np.log10(error1_list), label='Errors1')
        plt.plot(GN_steps_output_list, np.log10(error_ratio3_list), label='Errors3 / Errors2')
        
        # Fit lines to compute slopes
        log_GN_steps = np.log10(GN_steps_output_list)
        log_error1 = np.log10(error1_list)
        log_error_ratio3 = np.log10(error_ratio3_list)
        coeffs1 = np.polyfit(log_GN_steps, log_error1, 1)
        coeffs3 = np.polyfit(log_GN_steps, log_error_ratio3, 1)
        fitted_line1 = np.polyval(coeffs1, log_GN_steps)
        fitted_line3 = np.polyval(coeffs3, log_GN_steps)
        
        plt.plot(GN_steps_output_list, fitted_line1, '--', label=f'Fit Line 1 (Slope: {coeffs1[0]:.2f})')
        plt.plot(GN_steps_output_list, fitted_line3, '--', label=f'Fit Line 3 (Slope: {coeffs3[0]:.2f})')
        
        plt.xlabel('Training Iterations (GN_steps)')
        plt.ylabel('log10(Error) or log10(Error Ratio)')
        plt.title('ConvergenceRate Verification')
        plt.legend()
        plt.xscale('log')
        plt.savefig(f'{save_path}/ConvergenceRate_Verification.png')
        plt.close()
    
        # Disable the profiler and print stats
        profiler.disable()
        profiler.print_stats(sort='cumtime')
    
        return rhomax
