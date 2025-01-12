import numpy as np
import jax.numpy as jnp
from jax import random

class ScaSML_full_history(object):
    '''Multilevel Picard Iteration calibrated GP for high dimensional semilinear PDE'''
    #all the vectors uses rows as index and columns as dimensions
    def __init__(self, equation, GP):
        '''
        Initialize the ScaSML parameters.
        
        Parameters:
            equation (Equation): An object representing the equation to be solved.
            GP (GaussianProcess Solver): An object of Gaussian Process Solver for PDE.
        '''
        # Initialize ScaSML parameters from the equation
        self.equation = equation
        self.sigma = equation.sigma
        self.mu = equation.mu
        equation.geometry()
        self.T = equation.T
        self.t0 = equation.t0
        self.n_input = equation.n_input
        self.n_output = equation.n_output
        self.GP = GP
        # Note: A potential way to accelerate the inference process is to use a discretized version of the Laplacian.
        self.evaluation_counter=0 # Number of evaluations
     
    def f(self, x_t, u_breve, z_breve):
        '''
        Generator function of ScaSML, representing the light and large version.
        
        Parameters:
            x_t (ndarray): Input data of shape (batch_size, n_input).
            u_breve (ndarray): approximated u_ture-u_hat.
            z_breve (ndarray): approximated gradient of u_breve.
        
        Returns:
            ndarray: The output of the generator function of shape (batch_size,).
        '''
        eq = self.equation
        # batch_size=x_t.shape[0]
        # self.evaluation_counter+=batch_size
        self.evaluation_counter+=1
        u_hat = self.GP.predict(x_t)
        grad_u_hat_x = self.GP.compute_gradient(x_t, u_hat)[:,:-1]
        # Calculate the values for the generator function
        val1 = eq.f(x_t, u_breve + u_hat, eq.sigma(x_t) * (grad_u_hat_x)+ z_breve)
        val2 = eq.f(x_t, u_hat, eq.sigma(x_t) * grad_u_hat_x)
        return val1 - val2 #light version
    
    def g(self, x_t):
        '''
        Terminal constraint function of ScaSML.
        
        Parameters:
            x_t (ndarray): Input data of shape (batch_size, n_input).
        
        Returns:
            ndarray: The output of the terminal constraint function of shape (batch_size,).
        '''
        eq = self.equation
        # batch_size=x_t.shape[0]
        # self.evaluation_counter+=batch_size
        self.evaluation_counter+=1
        u_hat = self.GP.predict(x_t)
        # tensor_x_t[:, -1] = self.T
        # Calculate the result of the terminal constraint function
        result = eq.g(x_t) - u_hat
        # if np.abs(result).any() > 0.5:
        #     print(f'g:{result}')
        return result[:,0]
    
    def approx_parameters(self, rhomax):
        '''
        Approximates parameters for the multilevel Picard iteration.
        
        Args:
            rhomax (int): Maximum level of refinement.
                
        Returns:
            tuple: A tuple containing matrices for forward Euler steps (Mf), backward Euler steps (Mg).
                Shapes are as follows: Mf, Mg are (rhomax, rhomax).
        '''
        levels = list(range(1, rhomax + 1))  # Level list
        Mf = jnp.zeros((rhomax, rhomax), dtype=int)  # Initialize matrix for forward Euler steps
        Mg = jnp.zeros((rhomax, rhomax + 1), dtype=int)  # Initialize matrix for backward Euler steps
        for rho in range(1, rhomax + 1):
            for k in range(1, levels[rho - 1] + 1):
                Mf = Mf.at[rho - 1, k - 1].set(int(round(rho ** (k / 2))))  # Compute forward Euler steps
                Mg = Mg.at[rho - 1, k - 1].set(int(round(rho ** (k - 1))))  # Compute backward Euler steps
            Mg = Mg.at[rho - 1, rho].set(int(rho ** rho))  # Special case for backward Euler steps
        return Mf, Mg
    
    def set_approx_parameters(self, rhomax):
        '''
        Sets the approximation parameters for the multilevel Picard iteration.
        This method should be called before solving the PDE.
        
        Args:
            rhomax (int): Maximum level of refinement.
        '''
        self.Mf, self.Mg = self.approx_parameters(rhomax)  # Set approximation parameters
        
    # @log_variables
    def uz_solve(self, n, rho, x_t):
        '''
        Approximate the solution of the PDE, return the value of u(x_t) and z(x_t), batchwisely.
        
        Parameters:
            n (int): The index of summands in quadratic sum.
            rho (int): Number of levels of refinement.
            x_t (ndarray): A batch of spatial-temporal coordinates, shape (batch_size, n_input), where
                           batch_size is the number of samples in the batch and n_input is the number of input features (spatial dimensions + 1 for time).
        
        Returns:
            ndarray: The concatenated u and z values for each sample in the batch, shape (batch_size, 1+n_input-1).
                     Here, u is the approximate solution of the PDE at the given coordinates, and z is the associated spatial gradient.
        '''
        # Set alpha=1/2, beta=1/6
        # Extract model parameters and functions
        Mf, Mg = self.Mf, self.Mg
        T = self.T  # Terminal time
        dim = self.n_input - 1  # Spatial dimensions
        batch_size = x_t.shape[0]  # Batch size
        sigma = self.sigma(x_t)  # Volatility, scalar
        mu= self.mu(x_t)  # Drift, scalar
        x = x_t[:, :-1]  # Spatial coordinates, shape (batch_size, dim)
        t = x_t[:, -1]  # Temporal coordinates, shape (batch_size,)
        f = self.f  # Generator term function
        g = self.g  # Terminal constraint function
        
        # Manage random keys of JAX
        key = random.PRNGKey(0)  # Random key for generating Monte Carlo samples
        subkey = random.split(key, 1)[0]  # Subkey for generating Brownian increments
        
        # Determine the number of Monte Carlo samples for backward Euler
        MC = int(Mg[rho - 1, n])  # Number of Monte Carlo samples, scalar
        
        # Generate Monte Carlo samples for backward Euler
        std_normal = random.normal(subkey, shape=(batch_size, MC, dim))
        dW = jnp.sqrt(T-t)[:, np.newaxis, np.newaxis] * std_normal  # Brownian increments, shape (batch_size, MC, dim)
        self.evaluation_counter+=MC
        X = jnp.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)  # Replicated spatial coordinates, shape (batch_size, MC, dim)
        disturbed_X = X + mu*(T-t)[:, np.newaxis, np.newaxis]+ sigma * dW  # Disturbed spatial coordinates, shape (batch_size, MC, dim)
        
        # Initialize arrays for terminal and difference values
        terminals = jnp.zeros((batch_size, MC, 1))  # Terminal values, shape (batch_size, MC, 1)
        differences = jnp.zeros((batch_size, MC, 1))  # Differences, shape (batch_size, MC, 1)
        
        # Prepare terminal inputs
        input_terminal = jnp.concatenate((X, jnp.full((batch_size, MC, 1), T)), axis=2)  # Shape (batch_size, MC, n_input)
        disturbed_input_terminal = jnp.concatenate((disturbed_X, jnp.full((batch_size, MC, 1), T)), axis=2)  # Shape (batch_size, MC, n_input)

        # Flatten inputs for vectorized function evaluation
        input_terminal_flat = input_terminal.reshape(-1, self.n_input)
        disturbed_input_terminal_flat = disturbed_input_terminal.reshape(-1, self.n_input)

        # Vectorized evaluation of g
        terminals_flat = g(input_terminal_flat)  # Evaluate terminal condition, shape (batch_size * MC, 1)
        differences_flat = g(disturbed_input_terminal_flat) - terminals_flat  # Evaluate disturbed terminal condition, shape (batch_size * MC, 1)

        # Reshape back to (batch_size, MC, 1)
        terminals = terminals_flat.reshape(batch_size, MC, 1)
        differences = differences_flat.reshape(batch_size, MC, 1)
        
        # Compute u and z values
        u = jnp.mean(differences + terminals, axis=1)  # Mean over Monte Carlo samples, shape (batch_size, 1)
        delta_t = (T - t + 1e-6)[:, jnp.newaxis]  # Avoid division by zero, shape (batch_size, 1)
        z = jnp.mean(differences * std_normal, axis=1) / (delta_t)  # Compute z values, shape (batch_size, dim)           
        cated_uz = jnp.concatenate((u, z), axis=-1)  # Concatenate u and z values, shape (batch_size, dim + 1)

        # Recursive call for n > 0
        if n == 0:
            batch_size=x_t.shape[0]
            u_hat = self.GP.predict(x_t)
            grad_u_hat_x = self.GP.compute_gradient(x_t, u_hat)[:, :-1]   
            initial_value= jnp.concatenate((u_hat, sigma* grad_u_hat_x), axis=-1)        
            return initial_value 
        elif n < 0:
            return jnp.zeros_like(cated_uz)  # Return zeros if n < 0
        
        # Recursive computation for n > 0
        for l in range(n):
            MC = int(Mf[rho - 1, n - l - 1])  # Number of Monte Carlo samples, scalar
            tau = random.uniform(subkey, shape=(batch_size, MC)) ** 2  # Sample a random variable tau in (0,1) with density rho(s) = 1/(2*sqrt(s))
            # Multiply tau by (T-t)
            sampled_time_steps = (tau * (T-t)[:, jnp.newaxis]).reshape((batch_size, MC, 1))  # Sample time steps, shape (batch_size, MC)
            X = jnp.repeat(x.reshape(x.shape[0], 1, x.shape[1]), MC, axis=1)  # Replicated spatial coordinates, shape (batch_size, MC, dim)
            simulated = jnp.zeros((batch_size, MC, dim + 1))  # Initialize array for simulated values, shape (batch_size, MC, dim + 1)
            std_normal = random.normal(subkey, shape=(batch_size, MC, dim))  # Generate standard normal samples
            dW =jnp.sqrt(sampled_time_steps) * std_normal  # Brownian increments for current time step, shape (batch_size, MC, dim)
            self.evaluation_counter+=MC*dim
            X += mu*(sampled_time_steps)+sigma * dW  # Update spatial coordinates
            co_solver_l = lambda X_t: self.uz_solve(n=l, rho=rho, x_t=X_t)  # Co-solver for level l
            co_solver_l_minus_1 = lambda X_t: self.uz_solve(n=l - 1, rho=rho, x_t=X_t)  # Co-solver for level l - 1
            # Compute Compute u and z values for current quadrature point using vmap
            input_intermediates = jnp.concatenate((X, sampled_time_steps), axis=2)  # Intermediate spatial-temporal coordinates, shape (batch_size, MC, n_input)
            input_intermediates_flat = input_intermediates.reshape(-1, self.n_input)
            simulated_flat = co_solver_l(input_intermediates_flat)
            simulated = simulated_flat.reshape(batch_size, MC, dim + 1)
            simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC, 1), simulated[:, :, 1:]  # Extract u and z values, shapes (batch_size, MC, 1) and (batch_size, MC, dim)
            # Flatten inputs for vectorized function evaluation and apply generator term function
            simulated_u_flat = simulated_u.reshape(-1, 1)
            simulated_z_flat = simulated_z.reshape(-1, dim)
            y_flat = f(input_intermediates_flat, simulated_u_flat, simulated_z_flat)  # Apply generator term function, shape (batch_size * MC, 1)
            y = y_flat.reshape(batch_size, MC, 1)  # Reshape to shape (batch_size, MC, 1)
            # Update u and z values
            u += 2*(T-t)[:,jnp.newaxis]* jnp.mean(jnp.sqrt(tau)[:,:,jnp.newaxis]*y, axis=1)  # Update u values
            delta_t = (sampled_time_steps + 1e-6)  # Avoid division by zero, shape (batch_size, 1)
            z += 2*(T-t)[:,jnp.newaxis] * jnp.mean((np.sqrt(tau)[:,:,jnp.newaxis]*y * std_normal / (delta_t)),axis=1)  # Update z values                
            # Adjust u and z values if l > 0
            if l:
                # Compute Compute u and z values for current quadrature point using vmap
                input_intermediates = jnp.concatenate((X, sampled_time_steps), axis=2)  # Intermediate spatial-temporal coordinates, shape (batch_size, MC, n_input)
                input_intermediates_flat = input_intermediates.reshape(-1, self.n_input)
                simulated_flat = co_solver_l_minus_1(input_intermediates_flat)
                simulated = simulated_flat.reshape(batch_size, MC, dim + 1)
                simulated_u, simulated_z = simulated[:, :, 0].reshape(batch_size, MC, 1), simulated[:, :, 1:]  # Extract u and z values, shapes (batch_size, MC, 1) and (batch_size, MC, dim)
                # Flatten inputs for vectorized function evaluation and apply generator term function
                simulated_u_flat = simulated_u.reshape(-1, 1)
                simulated_z_flat = simulated_z.reshape(-1, dim)
                y_flat = f(input_intermediates_flat, simulated_u_flat, simulated_z_flat) # Apply generator term function, shape (batch_size * MC, 1)
                y = y_flat.reshape(batch_size, MC, 1)  # Reshape to shape (batch_size, MC, 1)
                # Update u and z values
                u -= 2*(T-t)[:,jnp.newaxis]* jnp.mean(jnp.sqrt(tau)[:,:,jnp.newaxis]*y, axis=1)  # Update u values
                delta_t = (sampled_time_steps + 1e-6)  # Avoid division by zero, shape (batch_size, 1)
                z -= 2*(T-t)[:,jnp.newaxis] * jnp.mean((jnp.sqrt(tau)[:,:,jnp.newaxis]*y * std_normal / (delta_t)),axis=1)  # Update z values  
        return jnp.concatenate((u, z), axis=-1)  # Concatenate adjusted u and z values, shape (batch_size, dim + 1)

    def u_solve(self, n, rho, x_t):
        '''
        Approximate the solution of the PDE, return the ndarray of u(x_t) only.
        
        Parameters:
            n (int): Index of summands in quadratic sum.
            rho (int): The current level.
            x_t (ndarray): A batch of spatial-temporal coordinates, shape (batch_size, n_input).
            
        Returns:
            ndarray: The u values, shape (batch_size,1).
        '''
        eq = self.equation
        # Calculate u_breve and z_breve using uz_solve
        u_breve_z_breve = self.uz_solve(n, rho, x_t)
        u_breve, z_breve = u_breve_z_breve[:, 0], u_breve_z_breve[:, 1:]
        
        batch_size=x_t.shape[0]
        u_hat = self.GP.predict(x_t)
        
        # Calculate and return the final u value, if |u_breve|<sqrt(compute_PDE_loss(x_t)), return u_breve+u_hat, else return u_hat
        uncertainty = jnp.sqrt(jnp.abs(self.GP.compute_PDE_loss(x_t)))
        u = jnp.where(jnp.abs(u_breve) < uncertainty, u_breve + u_hat, u_hat)
        return u