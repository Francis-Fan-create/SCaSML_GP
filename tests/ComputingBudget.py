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
import jax.numpy as jnp
from scipy import stats
import matplotlib.ticker as ticker

class ComputingBudget(object):
    '''
    Computing Budget test: compares solvers under equal total computational budget.
    
    This test demonstrates that SCaSML achieves better accuracy than MLP 
    when all methods are given the same total computing time (training + inference).
    
    For GP-based solvers:
    - GP: Trains with varying GN_steps to control computational budget
    - MLP: Uses different rho levels (quadrature points) for varying budgets
    - SCaSML: Combines GP training with varying rho levels

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
        Initializes the computing budget test with given solvers and equation.

        Parameters:
        equation (object): The equation object containing problem specifics.
        solver1 (object): The GP solver.
        solver2 (object): The MLP solver object.
        solver3 (object): The ScaSML solver object.
        '''
        # Save original stdout and stderr
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        # Initialize the parameters
        self.equation = equation
        self.dim = equation.n_input - 1
        self.solver1 = solver1
        self.solver2 = solver2
        self.solver3 = solver3
        self.t0 = equation.t0
        self.T = equation.T


    def test(self, save_path, budget_levels=[1, 2, 3, 4, 5], num_domain=1000, num_boundary=200):
        '''
        Compares solvers under different computing budget levels.
        
        The budget is controlled by:
        - GP: Number of GN_steps (Gauss-Newton iterations)
        - MLP: rho parameter (quadrature points) 
        - SCaSML: Combination of GP GN_steps and rho parameter
        
        Parameters:
        save_path (str): The path to save the results.
        budget_levels (list): List of budget levels (e.g., [1, 2, 3, 4, 5]).
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
        d = eq.n_input - 1
        
        # Fixed training data size
        num_train_domain = 1000
        num_train_boundary = 200
        data_domain_train, data_boundary_train = eq.generate_data(num_train_domain, num_train_boundary)
        
        # Generate test data (fixed across all budget levels)
        data_domain_test, data_boundary_test = eq.generate_test_data(num_domain, num_boundary)
        xt_test = np.concatenate((data_domain_test, data_boundary_test), axis=0)
        exact_sol = eq.exact_solution(xt_test)
        
        # Storage for results
        gp_errors = []
        mlp_errors = []
        scasml_errors = []
        gp_times = []
        mlp_times = []
        scasml_times = []
        # store statistics for printing in the final log
        gp_mean_list = []
        gp_std_list = []
        gp_ci_lower_list = []
        gp_ci_upper_list = []
        mlp_mean_list = []
        mlp_std_list = []
        mlp_ci_lower_list = []
        mlp_ci_upper_list = []
        scasml_mean_list = []
        scasml_std_list = []
        scasml_ci_lower_list = []
        scasml_ci_upper_list = []
        
        # Base parameters for budget=1
        base_gn_steps = 5
        base_rho = 2
        
        for budget in budget_levels:
            print(f"\n{'='*60}")
            print(f"Testing Budget Level: {budget}")
            print(f"{'='*60}")
            
            # Calculate parameters for this budget
            gn_steps = base_gn_steps * budget
            rho = base_rho + budget - 1
            
            # ==========================================
            # GP: Train and measure time
            # ==========================================
            print(f"\nTraining GP with GN_steps={gn_steps}...")
            solver1_copy = copy.deepcopy(self.solver1)
            
            start_time = time.time()
            solver1_copy.GPsolver(data_domain_train, data_boundary_train, GN_steps=gn_steps)
            train_time_gp = time.time() - start_time
            
            start_time = time.time()
            sol_gp = solver1_copy.predict(xt_test)
            inference_time_gp = time.time() - start_time
            
            total_time_gp = train_time_gp + inference_time_gp
            print(f"GP training time: {train_time_gp:.2f}s, inference time: {inference_time_gp:.2f}s, total: {total_time_gp:.2f}s")
            
            # ==========================================
            # MLP: Adjust rho to approximate budget
            # ==========================================
            print(f"\nRunning MLP with rho={rho}...")
            solver2_copy = copy.deepcopy(self.solver2)
            
            start_time = time.time()
            sol_mlp = solver2_copy.u_solve(rho, rho, xt_test)
            total_time_mlp = time.time() - start_time
            print(f"MLP total time: {total_time_mlp:.2f}s")
            
            # ==========================================
            # ScaSML: Combine GP training with rho
            # ==========================================
            print(f"\nRunning SCaSML with GN_steps={max(1, gn_steps//2)} and rho={rho}...")
            solver3_copy = copy.deepcopy(self.solver3)
            
            # Train GP backbone with fewer steps
            scasml_gn_steps = max(1, gn_steps // 2)
            start_time = time.time()
            solver3_copy.GP.GPsolver(data_domain_train, data_boundary_train, GN_steps=scasml_gn_steps)
            train_time_scasml = time.time() - start_time
            
            start_time = time.time()
            sol_scasml = solver3_copy.u_solve(rho, rho, xt_test)
            inference_time_scasml = time.time() - start_time
            
            total_time_scasml = train_time_scasml + inference_time_scasml
            print(f"SCaSML training time: {train_time_scasml:.2f}s, inference time: {inference_time_scasml:.2f}s, total: {total_time_scasml:.2f}s")
            
            # ==========================================
            # Compute Errors
            # ==========================================
            valid_mask = ~(np.isnan(sol_gp) | np.isnan(sol_mlp) | 
                          np.isnan(sol_scasml) | np.isnan(exact_sol)).flatten()
            
            if np.sum(valid_mask) == 0:
                print("Warning: All predictions are NaN. Skipping this budget level.")
                continue
            
            errors_gp = np.abs(sol_gp.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
            errors_mlp = np.abs(sol_mlp.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
            errors_scasml = np.abs(sol_scasml.flatten()[valid_mask] - exact_sol.flatten()[valid_mask])
            
            exact_sol_valid = exact_sol.flatten()[valid_mask]
            
            rel_error_gp = np.linalg.norm(errors_gp) / np.linalg.norm(exact_sol_valid)
            rel_error_mlp = np.linalg.norm(errors_mlp) / np.linalg.norm(exact_sol_valid)
            rel_error_scasml = np.linalg.norm(errors_scasml) / np.linalg.norm(exact_sol_valid)
            
            print(f"\nErrors for budget {budget}:")
            print(f"  GP relative L2 error: {rel_error_gp:.6e}")
            print(f"  MLP relative L2 error: {rel_error_mlp:.6e}")
            print(f"  SCaSML relative L2 error: {rel_error_scasml:.6e}")
            
            # Store results
            gp_errors.append(rel_error_gp)
            mlp_errors.append(rel_error_mlp)
            scasml_errors.append(rel_error_scasml)
            gp_times.append(total_time_gp)
            mlp_times.append(total_time_mlp)
            scasml_times.append(total_time_scasml)
            gp_mean_list.append(gp_mean)
            gp_std_list.append(gp_std)
            gp_ci_lower_list.append(ci_lower_gp)
            gp_ci_upper_list.append(ci_upper_gp)
            mlp_mean_list.append(mlp_mean)
            mlp_std_list.append(mlp_std)
            mlp_ci_lower_list.append(ci_lower_mlp)
            mlp_ci_upper_list.append(ci_upper_mlp)
            scasml_mean_list.append(scasml_mean)
            scasml_std_list.append(scasml_std)
            scasml_ci_lower_list.append(ci_lower_scasml)
            scasml_ci_upper_list.append(ci_upper_scasml)
            
            # ==========================================
            # Statistical Analysis
            # ==========================================
            # Calculate statistics
            gp_mean = np.mean(errors_gp)
            gp_std = np.std(errors_gp)
            mlp_mean = np.mean(errors_mlp)
            mlp_std = np.std(errors_mlp)
            scasml_mean = np.mean(errors_scasml)
            scasml_std = np.std(errors_scasml)
            
            # Confidence intervals
            def compute_ci_95(errors):
                mean = np.mean(errors)
                std = np.std(errors)
                n = len(errors)
                ci = 1.96 * std / np.sqrt(n)
                return mean - ci, mean + ci
            
            ci_lower_gp, ci_upper_gp = compute_ci_95(errors_gp)
            ci_lower_mlp, ci_upper_mlp = compute_ci_95(errors_mlp)
            ci_lower_scasml, ci_upper_scasml = compute_ci_95(errors_scasml)
            
            # Paired t-tests
            t_gp_scasml, p_gp_scasml = stats.ttest_rel(errors_gp, errors_scasml)
            t_mlp_scasml, p_mlp_scasml = stats.ttest_rel(errors_mlp, errors_scasml)
            
            # Improvement percentages
            improvement_gp = (rel_error_gp - rel_error_scasml) / rel_error_gp * 100
            improvement_mlp = (rel_error_mlp - rel_error_scasml) / rel_error_mlp * 100
            
            print(f"\nStatistical Analysis:")
            print(f"  GP improvement: {improvement_gp:+.2f}% (p-value: {p_gp_scasml:.4f})")
            print(f"  MLP improvement: {improvement_mlp:+.2f}% (p-value: {p_mlp_scasml:.4f})")
            
            # Log to wandb
            wandb.log({
                f"budget_{budget}_gp_error": rel_error_gp,
                f"budget_{budget}_mlp_error": rel_error_mlp,
                f"budget_{budget}_scasml_error": rel_error_scasml,
                f"budget_{budget}_gp_time": total_time_gp,
                f"budget_{budget}_mlp_time": total_time_mlp,
                f"budget_{budget}_scasml_time": total_time_scasml,
                f"budget_{budget}_improvement_vs_gp": improvement_gp,
                f"budget_{budget}_improvement_vs_mlp": improvement_mlp,
                f"budget_{budget}_p_gp_scasml": p_gp_scasml,
                f"budget_{budget}_p_mlp_scasml": p_mlp_scasml,
                f"budget_{budget}_gp_mean": gp_mean,
                f"budget_{budget}_gp_std": gp_std,
                f"budget_{budget}_mlp_mean": mlp_mean,
                f"budget_{budget}_mlp_std": mlp_std,
                f"budget_{budget}_scasml_mean": scasml_mean,
                f"budget_{budget}_scasml_std": scasml_std,
            })
        
        # ==========================================
        # Visualization
        # ==========================================
        # Color scheme
        COLOR_SCHEME = {
            'GP': '#000000',       # Black
            'MLP': '#A6A3A4',      # Gray
            'SCaSML': '#2C939A'    # Teal
        }
        
        # Configure matplotlib
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'axes.labelsize': 10,
            'axes.titlesize': 0,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.linewidth': 0.8,
            'lines.linewidth': 1.2,
            'lines.markersize': 5,
            'savefig.dpi': 600,
            'savefig.transparent': True,
            'figure.autolayout': False
        })
        
        budget_array = np.array(budget_levels[:len(gp_errors)])
        
        # ==========================================
        # Figure 1: Error vs Budget
        # ==========================================
        fig, ax = plt.subplots(figsize=(3.5, 3))
        
        # Plot with markers
        ax.plot(budget_array, gp_errors, color=COLOR_SCHEME['GP'], 
               marker='o', linestyle='-', label='GP', 
               markerfacecolor='none', markeredgewidth=0.8)
        ax.plot(budget_array, mlp_errors, color=COLOR_SCHEME['MLP'], 
               marker='s', linestyle='-', label='MLP',
               markerfacecolor='none', markeredgewidth=0.8)
        ax.plot(budget_array, scasml_errors, color=COLOR_SCHEME['SCaSML'], 
               marker='^', linestyle='-', label='SCaSML',
               markerfacecolor='none', markeredgewidth=0.8)
        
        ax.set_xlabel('Computing Budget (×baseline)', labelpad=3)
        ax.set_ylabel('Relative L2 Error', labelpad=3)
        ax.set_yscale('log')
        # Keep x-axis scale consistent with the old implementation
        ax.set_xscale('log')
        ax.legend(frameon=False, loc='upper right')
        ax.grid(True, which='major', axis='both', linestyle='--', 
               linewidth=0.5, alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/Error_vs_Budget.pdf', 
                   bbox_inches='tight', pad_inches=0.05)
        plt.close()
        
        # ==========================================
        # Figure 2: Improvement Bar Chart
        # ==========================================
        fig, ax = plt.subplots(figsize=(3.5, 3))
        
        # Calculate improvements at each budget level
        improvements_vs_gp = [(gp - scasml) / gp * 100 
                               for gp, scasml in zip(gp_errors, scasml_errors)]
        improvements_vs_mlp = [(mlp - scasml) / mlp * 100 
                              for mlp, scasml in zip(mlp_errors, scasml_errors)]
        
        x = np.arange(len(budget_array))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, improvements_vs_gp, width, 
                      label='SCaSML vs GP', color=COLOR_SCHEME['GP'], 
                      edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, improvements_vs_mlp, width, 
                      label='SCaSML vs MLP', color=COLOR_SCHEME['MLP'], 
                      edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Computing Budget (×baseline)', labelpad=3)
        ax.set_ylabel('Improvement (%)', labelpad=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b}×' for b in budget_array])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.legend(frameon=False, loc='upper left')
        ax.grid(True, which='major', axis='y', linestyle='--', 
               linewidth=0.5, alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/Improvement_Bar_Chart.pdf', 
                   bbox_inches='tight', pad_inches=0.05)
        plt.close()
        
        
        # ==========================================
        # Summary Statistics and Final Log Output
        # ==========================================
        avg_improvement_gp = np.mean(improvements_vs_gp)
        avg_improvement_mlp = np.mean(improvements_vs_mlp)
        
        wandb.log({
            "avg_improvement_vs_gp": avg_improvement_gp,
            "avg_improvement_vs_mlp": avg_improvement_mlp,
        })
        
        # Write final results to log file
        log_file = open(f"{save_path}/ComputingBudget.log", "w")
        sys.stdout = log_file
        sys.stderr = log_file
        
        print("=" * 80)
        print("COMPUTING BUDGET TEST - FINAL RESULTS")
        print("=" * 80)
        print(f"Equation: {eq_name}")
        print(f"Dimension: {d+1}")
        print(f"Budget levels tested: {budget_array.tolist()}")
        print("=" * 80)
        print()
        
        print(f"{'Budget':<12} {'GP Error':<15} {'MLP Error':<15} {'SCaSML Error':<15}")
        print("-" * 60)
        for i, budget in enumerate(budget_array):
            print(f"{budget:<12.0f} {gp_errors[i]:<15.6e} {mlp_errors[i]:<15.6e} {scasml_errors[i]:<15.6e}")
        print()

        # Print summary statistics (mean, std and 95% CI) for each budget
        print("Detailed mean/std/95%CI per budget:")
        print(f"{'Budget':<12} {'GP mean(std) [95% CI]':<35} {'MLP mean(std) [95% CI]':<35} {'SCaSML mean(std) [95% CI]':<35}")
        print("-" * 120)
        for i, budget in enumerate(budget_array):
            print(f"{budget:<12.0f} {gp_mean_list[i]:<12.6e}({gp_std_list[i]:.2e}) [{gp_ci_lower_list[i]:.2e}, {gp_ci_upper_list[i]:.2e}] {mlp_mean_list[i]:<12.6e}({mlp_std_list[i]:.2e}) [{mlp_ci_lower_list[i]:.2e}, {mlp_ci_upper_list[i]:.2e}] {scasml_mean_list[i]:<12.6e}({scasml_std_list[i]:.2e}) [{scasml_ci_lower_list[i]:.2e}, {scasml_ci_upper_list[i]:.2e}]")
        print()
        
        print(f"{'Budget':<12} {'GP Time':<15} {'MLP Time':<15} {'SCaSML Time':<15}")
        print("-" * 60)
        for i, budget in enumerate(budget_array):
            print(f"{budget:<12.0f} {gp_times[i]:<15.2f} {mlp_times[i]:<15.2f} {scasml_times[i]:<15.2f}")
        print()
        
        print("Average Improvement:")
        print(f"  SCaSML vs GP: {avg_improvement_gp:+.2f}%")
        print(f"  SCaSML vs MLP: {avg_improvement_mlp:+.2f}%")
        print()
        
        print(f"Final Budget Level ({budget_array[-1]:.0f}×):")
        print(f"  GP error: {gp_errors[-1]:.6e}")
        print(f"  MLP error: {mlp_errors[-1]:.6e}")
        print(f"  SCaSML error: {scasml_errors[-1]:.6e}")
        print(f"  Improvement vs GP: {improvements_vs_gp[-1]:+.2f}%")
        print(f"  Improvement vs MLP: {improvements_vs_mlp[-1]:+.2f}%")
        print("=" * 80)
        
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        log_file.close()
        
        # Stop profiler
        profiler.disable()
        profiler.dump_stats(f"{save_path}/{eq_name}_computing_budget.prof")
        
        # Upload profiler results
        artifact = wandb.Artifact(f"{eq_name}_computing_budget", type="profile")
        artifact.add_file(f"{save_path}/{eq_name}_computing_budget.prof")
        wandb.log_artifact(artifact)
        
        print("Computing budget test completed!")
        return budget_levels[-1] if budget_levels else 1

