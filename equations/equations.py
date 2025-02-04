import deepxde as dde
# import numpy as np
import torch
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial

import sys
import os
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from solvers.MLP import MLP # use MLP to deal with equations without explicit solutions

class Equation(object):
    '''Equation class for PDEs based on deepxde framework'''
    # all the vectors use rows as index and columns as dimensions
    
    def __init__(self, n_input, n_output=1):
        """
        Initialize the equation parameters.
        We assume that u is a scalar if n_output is not specified.
        
        Args:
            n_input (int): Dimension of the input, including time.
            n_output (int, optional): Dimension of the output. Defaults to 1.
        """
        self.n_input = n_input  # dimension of the input, including time
        self.n_output = n_output  # dimension of the output

    def PDE_loss(self, x_t, u, z):
        """
        PINN loss in the PDE, used in ScaSML to calculate epsilon.
        
        Args:
            x_t (tensor): The input data, shape (n_samples, n_input).
            u (tensor): The solution, shape (n_samples, n_output).
            z (tensor): The gradient of u w.r.t. x, shape (n_samples, n_input-1).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def gPDE_loss(self, x_t, u):
        """
        gPINN loss in the PDE, used for training.
        
        Args:
            x_t (tensor): The input data, shape (n_samples, n_input).
            u (tensor): The solution, shape (n_samples, n_output).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def terminal_constraint(self, x_t):
        """
        Terminal constraint in the PDE.
        
        Args:
            x_t (ndarray): The input data at terminal time, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
    def initial_constraint(self, x_t):
        """
        Initial constraint in the PDE.
        
        Args:
            x_t (ndarray): The input data at terminal time, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError    

    def Dirichlet_boundary_constraint(self, x_t):
        """
        Dirichlet boundary constraint in the PDE.
        
        Args:
            x_t (ndarray): The input data at the boundary, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def Neumann_boundary_constraint(self, x_t):
        """
        Neumann boundary constraint in the PDE.
        
        Args:
            x_t (ndarray): The input data at terminal time, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
    def mu(self, x_t):
        """
        Drift coefficient in PDE, usually a vector.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def sigma(self, x_t):
        """
        Diffusion coefficient in PDE, usually a matrix.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def f(self, x_t, u, z):
        """
        Generator term in PDE, usually a vector.
        z is the product of the gradient of u and sigma, usually a vector, since u is a scalar.
        Note that z does not include the time dimension.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            u (ndarray): The solution, shape (n_samples, n_output).
            z (ndarray): The gradient of u w.r.t. x, shape (n_samples, n_input-1).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def g(self, x_t):
        """
        Terminal constraint in PDE, usually a vector.
        
        Args:
            x_t (ndarray): The input data at terminal time, shape (n_samples, n_input).
            
        Returns:
            ndarray: The terminal constraint value, shape (n_samples, n_output).
            
        Raises:
            NotImplementedError: If the terminal_constraint method is not implemented.
        """
        if hasattr(self, 'terminal_constraint'):
            return self.terminal_constraint(x_t)
        else:
            raise NotImplementedError

    def exact_solution(self, x_t):
        """
        Exact solution of the PDE, which will not be used in the training, but for testing.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

    def data_loss(self, x_t):
        """
        Data loss in PDE.
        
        Args:
            x_t (ndarray): The input data, shape (n_samples, n_input).
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
    def geometry(self, t0, T):
        """
        Geometry of the train domain.
        
        Args:
            t0 (float): The initial time.
            T (float): The terminal time.
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
    def test_geometry(self, t0, T):
        """
        Geometry of the test domain.
        
        Args:
            t0 (float): The initial time.
            T (float): The terminal time.
            
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
    def generate_data(self):
        """
        Generate data for training.
        
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError
    
    def generate_test_data(self):
        """
        Generate data for testing.
        
        Raises:
            NotImplementedError: This is a placeholder method.
        """
        raise NotImplementedError

class Grad_Dependent_Nonlinear(Equation):
    '''
    Example of a high-dimensional PDE with an exact solution.
    '''
    def __init__(self, n_input, n_output=1):
        '''
        Initializes the PDE example with specified input and output dimensions.
        
        Parameters:
        - n_input (int): The dimension of the input space, including the time dimension.
        - n_output (int): The dimension of the output space. Defaults to 1.
        '''
        super().__init__(n_input, n_output)
        self.uncertainty = 1e-1
        self.norm_estimation = 1
    
    @partial(jit,static_argnames=["self"])
    def terminal_constraint(self, x_t):
        '''
        Defines the terminal constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the terminal constraint.
        '''
        result = 1 - 1 / (1 + jnp.exp(x_t[:, -1] + jnp.sum(x_t[:, :self.n_input-1], axis=1)))
        result = result[:, jnp.newaxis]  # Convert to 2D
        return result 
    
    def mu(self, x_t=0):
        '''
        Returns the drift coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The drift coefficient.
        '''
        sigma = self.sigma()
        d = self.n_input - 1
        result = -1/(d * sigma) - sigma/ 2
        return result
    
    def sigma(self, x_t=0):
        '''
        Returns the diffusion coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The diffusion coefficient.
        '''
        return 0.25
    
    @partial(jit,static_argnames=["self"])
    def f(self, x_t, u, z):
        '''
        Defines the generator term for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (ndarray): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        - z (ndarray): Tensor of shape (batch_size, n_input-1), representing gradients.
        
        Returns:
        - result (ndarray): A 2D array of shape (batch_size, n_output), representing the generator term.
        '''
        result = self.sigma() * u * jnp.sum(z, axis=1, keepdims=True)
        return result
    
    @partial(jit,static_argnames=["self"])
    def exact_solution(self, x_t):
        '''
        Computes the exact solution of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An array of shape (batch_size, 1), representing the exact solution.
        '''
        t = x_t[:, -1]
        x = x_t[:, :-1]
        sum_x = jnp.sum(x, axis=1)
        exp_term = jnp.exp(t + sum_x)  # Computes the exponential term of the solution.
        result = 1 - 1 / (1 + exp_term)  # Computes the exact solution.
        result = result[:, jnp.newaxis]  # Convert to 2D
        return result
    
    @partial(jit,static_argnames=["self"])
    def exact_solution_derivative(self, x_t):
        '''
        Computes the exact solution derivative of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An array of shape (batch_size, n_input-1), representing the exact solution derivative.
        '''
        t = x_t[:, -1]
        x = x_t[:, :-1]
        sum_x = jnp.sum(x, axis=1)
        exp_term = jnp.exp(t + sum_x)  # Computes the exponential term of the solution.
        result = exp_term / (1 + exp_term)**2  # Computes the exact solution derivative.
        result = result[:, jnp.newaxis]  # Convert to 2D
        return result
    
    def geometry(self, t0=0, T=0.5):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0 = t0
        self.T = T
        self.radius = 0.5
        spacedomain = dde.geometry.Hypercube([-self.radius] * (self.n_input - 1), [self.radius] * (self.n_input - 1))  # Defines the spatial domain, for train
        timedomain = dde.geometry.TimeDomain(t0, self.T)  # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)  # Combines spatial and time domains.
        self.geomx = spacedomain
        self.geomt = timedomain
        return geom
    
    def test_geometry(self, t0=0, T=0.5):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0 = t0
        self.T = T
        self.test_T = T
        self.test_radius = 0.5
        spacedomain = dde.geometry.Hypercube([-self.test_radius] * (self.n_input - 1), [self.test_radius] * (self.n_input - 1))  # Defines the spatial domain, for test.
        timedomain = dde.geometry.TimeDomain(t0, self.test_T)  # Defines the time domain for test.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)  # Combines spatial and time domains.
        self.geomx = spacedomain
        self.geomt = timedomain
        return geom    
    
    def generate_data(self, num_domain=100, num_boundary=20):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        - num_boundary (int): Number of points to sample on the boundary.
        
        Returns:
        - data (tuple): A tuple containing domain points and boundary points.
        '''
        geom = self.geometry()  # Defines the geometry of the domain.
        data1 = geom.random_points(num_domain)  # Generates random points in the domain.
        data2 = geom.random_boundary_points(num_boundary)  # Generates random points on the boundary.
        return data1, data2

    def generate_test_data(self, num_domain=100, num_boundary=20):
        '''
        Generates data for testing the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        - num_boundary (int): Number of points to sample on the boundary.
        
        Returns:
        - data (tuple): A tuple containing domain points and boundary points.
        '''
        geom = self.test_geometry()  # Defines the geometry of the domain.
        data1 = geom.random_points(num_domain)  # Generates random points in the domain.
        data2 = geom.random_boundary_points(num_boundary)  # Generates random points on the boundary.
        return data1, data2
    
class Linear_HJB(Equation):
    '''
    Complicated HJB equation.
    '''
    def __init__(self, n_input, n_output=1):
        '''
        Initializes the PDE example with specified input and output dimensions.
        
        Parameters:
        - n_input (int): The dimension of the input space, including the time dimension.
        - n_output (int): The dimension of the output space. Defaults to 1.
        '''
        super().__init__(n_input, n_output)
        self.uncertainty = 1
        self.norm_estimation = 100
    
    @partial(jit,static_argnames=["self"])
    def terminal_constraint(self, x_t):
        '''
        Defines the terminal constraint for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): A 2D tensor of shape (batch_size, 1), representing the terminal constraint.
        '''
        x = x_t[:, :self.n_input - 1]  # Extracts the spatial coordinates.
        sum_x = jnp.sum(x, axis=1)  # Computes the sum of spatial coordinates.
        result = sum_x  # Computes the terminal constraint.
        result = result[:, jnp.newaxis]  # Convert to 2D
        return result 
    
    def mu(self, x_t=0):
        '''
        Returns the drift coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The drift coefficient.
        '''
        d = self.n_input - 1
        return -1/d
    
    def sigma(self, x_t=0):
        '''
        Returns the diffusion coefficient of the PDE. Here, it's a constant value.
        
        Parameters:
        - x_t (int, optional): Not used in this implementation.
        
        Returns:
        - (float): The diffusion coefficient.
        '''
        return jnp.sqrt(2)
    
    @partial(jit,static_argnames=["self"])
    def f(self, x_t, u, z):
        '''
        Defines the generator term for the PDE.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        - u (ndarray): Output tensor of shape (batch_size, n_output), representing the solution of the PDE.
        - z (ndarray): Tensor of shape (batch_size, n_input-1), representing gradients.
        
        Returns:
        - result (ndarray): A 2D array of shape (batch_size, n_output), representing the generator term.
        '''
        return 2 * jnp.ones_like(u)  # Shape: (batch_size, n_output)
    
    @partial(jit,static_argnames=["self"])
    def exact_solution(self, x_t):
        '''
        Computes the exact solution of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An array of shape (batch_size, 1), representing the exact solution.
        '''
        t = x_t[:, -1]
        x = x_t[:, :-1]
        sum_x = jnp.sum(x, axis=1)
        result = sum_x + (self.T - t)
        result = result[:, jnp.newaxis]  # Convert to 2D
        return result
    
    @partial(jit,static_argnames=["self"])
    def exact_solution_derivative(self, x_t):
        '''
        Computes the exact solution derivative of the PDE for given inputs.
        
        Parameters:
        - x_t (ndarray): Input tensor of shape (batch_size, n_input), where n_input includes the time dimension.
        
        Returns:
        - result (ndarray): An array of shape (batch_size, n_input-1), representing the exact solution derivative.
        '''
        result = jnp.ones((x_t.shape[0], self.n_input - 1))
        return result
    
    def geometry(self, t0=0, T=0.5):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0 = t0
        self.T = T
        self.radius = 0.5
        spacedomain = dde.geometry.Hypercube([-self.radius] * (self.n_input - 1), [self.radius] * (self.n_input - 1))  # Defines the spatial domain, for train
        timedomain = dde.geometry.TimeDomain(t0, self.T)  # Defines the time domain.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)  # Combines spatial and time domains.
        self.geomx = spacedomain
        self.geomt = timedomain
        return geom
    
    def test_geometry(self, t0=0, T=0.5):
        '''
        Defines the geometry of the domain for the PDE.
        
        Parameters:
        - t0 (float): Initial time.
        - T (float): Terminal time.
        
        Returns:
        - geom (dde.geometry.GeometryXTime): A GeometryXTime object representing the domain.
        '''
        self.t0 = t0
        self.T = T
        self.test_T = T
        self.test_radius = 0.5
        spacedomain = dde.geometry.Hypercube([-self.test_radius] * (self.n_input - 1), [self.test_radius] * (self.n_input - 1))  # Defines the spatial domain, for test.
        timedomain = dde.geometry.TimeDomain(t0, self.test_T)  # Defines the time domain for test.
        geom = dde.geometry.GeometryXTime(spacedomain, timedomain)  # Combines spatial and time domains.
        self.geomx = spacedomain
        self.geomt = timedomain
        return geom   
    
    def generate_data(self, num_domain=100, num_boundary=20):
        '''
        Generates data for training the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        - num_boundary (int): Number of points to sample on the boundary.
        
        Returns:
        - data (tuple): A tuple containing domain points and boundary points.
        '''
        geom = self.geometry()  # Defines the geometry of the domain.
        data1 = geom.random_points(num_domain)  # Generates random points in the domain.
        data2 = geom.random_boundary_points(num_boundary)  # Generates random points on the boundary.
        return data1, data2
    
    def generate_test_data(self, num_domain=100, num_boundary=20):
        '''
        Generates data for testing the PDE model.
        
        Parameters:
        - num_domain (int): Number of points to sample in the domain.
        - num_boundary (int): Number of points to sample on the boundary.
        
        Returns:
        - data (tuple): A tuple containing domain points and boundary points.
        '''
        geom = self.test_geometry()  # Defines the geometry of the domain.
        data1 = geom.random_points(num_domain)  # Generates random points in the domain.
        data2 = geom.random_boundary_points(num_boundary)  # Generates random points on the boundary.
        return data1, data2