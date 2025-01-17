import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import time
import sys
import os
import cProfile
import shutil
import jax.numpy as jnp

class SimpleUniform(object):
    '''
    Simple Uniform test in high dimensions.

    Attributes:
    equation (object): An object representing the equation to solve.
    dim (int): The dimension of the input space minus one.
    solver1 (object): A jax Gaussian Process model.
    solver2 (object): An object for the MLP solver.
    solver3 (object): An object for the ScaSML solver.
    t0 (float): The initial time.
    T (float): The final time.
    '''
    def __init__(self, equation, solver1, solver2, solver3):
        '''
        Initializes the simple uniform test with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The GP solver.
        solver2 (object): The MLP solver object.
        solver3 (object): The ScaSML solver object.
        '''
        #save original stdout and stderr
        self.stdout=sys.stdout
        self.stderr=sys.stderr
        # Initialize the parameters
        self.equation = equation
        self.dim = equation.n_input - 1  # equation.n_input: int
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.t0 = equation.t0  # equation.t0: float
        self.T = equation.T  # equation.T: float

    def test(self, save_path, rhomax=2, num_domain=500, num_boundary=100):
        '''
        Compares solvers on test data after training on a large training dataset.
    
        Parameters:
        save_path (str): The path to save the results.
        rhomax (int): The number of quadrature points for the approximation, equal to the total level
        num_domain (int): The number of points in the test domain.
        num_boundary (int): The number of points on the test boundary.
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
        n = rhomax
    
        # Generate large training data
        num_train_domain = 1000  # Large number for training domain data
        num_train_boundary = 200  # Large number for training boundary data
        data_domain_train, data_boundary_train = eq.generate_data(num_train_domain, num_train_boundary)
    
        # Train solver1 on large training data
        print("Training solver1 on large training dataset...")
        xt_domain_train = data_domain_train[:, :self.dim + 1]
        xt_boundary_train = data_boundary_train[:, :self.dim + 1]
        self.solver1.GPsolver(xt_domain_train, xt_boundary_train)
    
        # Generate test data
        data_domain_test, data_boundary_test = eq.generate_test_data(num_domain, num_boundary, random = "LHS")
        data_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
        xt_test = data_test[:, :self.dim + 1]
        exact_sol = eq.exact_solution(xt_test)
    
        errors1 = np.zeros(num_domain)
        errors2 = np.zeros(num_domain)
        errors3 = np.zeros(num_domain)
        rel_error1 = 0
        rel_error2 = 0
        rel_error3 = 0
        real_sol_L2 = 0
        time1, time2, time3 = 0, 0, 0
    
        # Measure the time and predict using solver1
        print("Predicting with solver1 on test data...")
        start = time.time()
        sol1 = self.solver1.predict(xt_test)
        time1 += time.time() - start
    
        # Measure the time and predict using solver2
        print("Predicting with solver2 on test data...")
        start = time.time()
        sol2 = self.solver2.u_solve(n, rhomax, data_test)
        time2 += time.time() - start
    
        # Measure the time and predict using solver3
        print("Predicting with solver3 on test data...")
        start = time.time()
        sol3 = self.solver3.u_solve(n, rhomax, data_test)
        time3 += time.time() - start

        # Compute the average error and relative error
        errors1 = np.abs(sol1 - exact_sol)
        errors2 = np.abs(sol2 - exact_sol)
        errors3 = np.abs(sol3 - exact_sol)
        rel_error1 = np.linalg.norm(errors1) / np.linalg.norm(exact_sol)
        rel_error2 = np.linalg.norm(errors2) / np.linalg.norm(exact_sol)
        rel_error3 = np.linalg.norm(errors3) / np.linalg.norm(exact_sol)
        real_sol_L2 = np.linalg.norm(exact_sol) / np.sqrt(exact_sol.shape[0])
        PDE_loss = self.solver1.compute_PDE_loss(data_test)
        #stop the profiler
        profiler.disable()
        #save the profiler results
        profiler.dump_stats(f"{save_path}/{eq_name}_rho_{rhomax}.prof")
        #upload the profiler results to wandb
        artifact=wandb.Artifact(f"{eq_name}_rho_{rhomax}", type="profile")
        artifact.add_file(f"{save_path}/{eq_name}_rho_{rhomax}.prof")
        wandb.log_artifact(artifact)
        # open a file to save the output
        log_file = open(f"{save_path}/SimpleUniform.log", "w")
        #redirect stdout and stderr to the log file
        sys.stdout=log_file
        sys.stderr=log_file
        # Print the total time for each solver
        print(f"Total time for GP: {time1} seconds")
        print(f"Total time for MLP: {time2} seconds")
        print(f"Total time for ScaSML: {time3} seconds")
        wandb.log({"Total time for GP": time1, "Total time for MLP": time2, "Total time for ScaSML": time3})
        # compute |errors1|-|errors3|,|errrors2|-|errors3|
        errors_13 = errors1 - errors3
        errors_23 = errors2 - errors3
        
        plt.figure()
        # collect all absolute errors
        # errors = [errors1.flatten(), errors2.flatten(), errors3.flatten(), errors_13.flatten(), errors_23.flatten()]
        errors = [errors1.flatten(), errors2.flatten(), errors3.flatten()]
        # Create a boxplot
        # plt.boxplot(errors, labels=['GP_l1', 'MLP_l1', 'ScaSML_l1', 'GP_l1 - ScaSML_l1', 'MLP_l1 - ScaSML_l1'])
        plt.boxplot(errors, labels=['GP_L2', 'MLP_L2', 'ScaSML_L2'])
        plt.xticks(rotation=45)
        plt.yscale('log')
        # Add a title and labels
        plt.title('L2 Error Distribution')
        plt.ylabel('L2 Error Value')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f"{save_path}/L2_Error_Distribution.png")
        # Upload the plot to wandb
        wandb.log({"Error Distribution": wandb.Image(f"{save_path}/L2_Error_Distribution.png")})
        
        plt.figure()
        # collect all absolute errors
        errors = [errors1.flatten(), errors2.flatten(), errors3.flatten()]
        # Calculate means and standard deviations
        means = [np.mean(e) for e in errors]
        stds = [np.std(e) for e in errors]
        # Define labels
        labels = ['GP_L2', 'MLP_L2', 'ScaSML_L2']
        x_pos = range(len(labels))
        # Create an error bar plot
        plt.errorbar(x_pos, means, yerr=stds, capsize=5, capthick=2, ecolor='black',  marker='s', markersize=7, mfc='red', mec='black')
        plt.xticks(x_pos, labels, rotation=45)
        plt.yscale('log')
        # Add a title and labels
        plt.title('L2 Error Distribution')
        plt.ylabel('L2 Error Value')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f"{save_path}/L2_Error_Distribution_errorbar.png")
        # Upload the plot to wandb
        wandb.log({"Error Distribution": wandb.Image(f"{save_path}/L2_Error_Distribution_errorbar.png")})

        # Print the results
        print(f"GP rel L2, rho={rhomax}->", rel_error1)
        
        
        print(f"MLP rel L2, rho={rhomax}->", rel_error2)
        
        
        print(f"ScaSML rel L2, rho={rhomax}->", rel_error3)
        
        
        print("Real Solution->", real_sol_L2)
        
        print(f"PDE Loss->", "min:", np.min(PDE_loss), "max:", np.max(PDE_loss), "mean:", np.mean(PDE_loss))

        print(f"GP L1, rho={rhomax}->","min:", np.min(errors1), "max:", np.max(errors1), "mean:", np.mean(errors1))
        
        
        print(f"MLP L1, rho={rhomax}->","min:", np.min(errors2), "max:", np.max(errors2), "mean:", np.mean(errors2))
        
        
        print(f"ScaSML L1, rho={rhomax}->","min:", np.min(errors3), "max:", np.max(errors3), "mean:", np.mean(errors3))
        
        
        # Calculate the sums of positive and negative differences
        positive_sum_13 = np.sum(errors_13[errors_13 > 0])
        negative_sum_13 = np.sum(errors_13[errors_13 < 0])
        positive_sum_23 = np.sum(errors_23[errors_23 > 0])
        negative_sum_23 = np.sum(errors_23[errors_23 < 0])
        # Display the positive count, negative count, positive sum, and negative sum of the difference of the errors
        print(f'GP L2 - ScaSML L2, rho={rhomax}->','positive count:', np.sum(errors_13 > 0), 'negative count:', np.sum(errors_13 < 0), 'positive sum:', positive_sum_13, 'negative sum:', negative_sum_13)
        print(f'MLP L2 - ScaSML L2, rho={rhomax}->','positive count:', np.sum(errors_23 > 0), 'negative count:', np.sum(errors_23 < 0), 'positive sum:', positive_sum_23, 'negative sum:', negative_sum_23)
        # Log the results to wandb
        wandb.log({f"mean of GP L2, rho={rhomax}": np.mean(errors1), f"mean of MLP L2, rho={rhomax}": np.mean(errors2), f"mean of ScaSML L2, rho={rhomax}": np.mean(errors3)})
        wandb.log({f"min of GP L2, rho={rhomax}": np.min(errors1), f"min of MLP L2, rho={rhomax}": np.min(errors2), f"min of ScaSML L2, rho={rhomax}": np.min(errors3)})
        wandb.log({f"max of GP L2, rho={rhomax}": np.max(errors1), f"max of MLP L2, rho={rhomax}": np.max(errors2), f"max of ScaSML L2, rho={rhomax}": np.max(errors3)})
        wandb.log({f"positive count of GP L2 - ScaSML L2, rho={rhomax}": np.sum(errors_13 > 0), f"negative count of GP L2 - ScaSML L2, rho={rhomax}": np.sum(errors_13 < 0), f"positive sum of GP L2 - ScaSML L2, rho={rhomax}": positive_sum_13, f"negative sum of GP L2 - ScaSML L2, rho={rhomax}": negative_sum_13})
        wandb.log({f"positive count of MLP L2 - ScaSML L2, rho={rhomax}": np.sum(errors_23 > 0), f"negative count of MLP L2 - ScaSML L2, rho={rhomax}": np.sum(errors_23 < 0), f"positive sum of MLP L2 - ScaSML L2, rho={rhomax}": positive_sum_23, f"negative sum of MLP L2 - ScaSML L2, rho={rhomax}": negative_sum_23})
        # reset stdout and stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        #close the log file
        log_file.close()
        return rhomax
