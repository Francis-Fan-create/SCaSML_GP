import torch
import numpy as np
import deepxde as dde
import torch.nn as nn
class GP(object):
    '''Gaussian Kernel Solver for a specific high dimensional PDE'''
    #since the explicit expression of f is required in the solver, we only use this solver for a specific PDE, named GP_Example
    #all the vectors uses rows as index and columns as dimensions
    def __init__(self, equation):
        '''
        Initialize the MLP parameters based on the given equation.
        
        Args:
            equation: An object containing the parameters and functions defining the equation to be solved by the MLP.
        '''
        # Initialize the MLP parameters from the equation object
        self.equation = equation
        equation.geometry()  # Initialize the geometry related parameters in the equation
        self.T = equation.T  # Terminal time
        self.t0 = equation.t0  # Initial time
        self.n_input = equation.n_input  # Number of input features
        self.n_output = equation.n_output  # Number of output features
        self.d= self.n_input-1 # Number of spatial dimensions
        self.sigma= 0.25*np.sqrt(self.d) # Standard deviation 
        self.nugget = 1e-10  # Regularization parameter to ensure numerical stability
        self.alpha= 1.0 # Optimization step size
        self.m=equation.m # Power in the equation
    
    def kappa(self, x, y, d, sigma):
        """
        Compute the Gaussian kernel function kappa.

        Args:
            x (np.ndarray): Coordinates of input point x.
            y (np.ndarray): Coordinates of input point y.
            d (int): Dimension of the input points.
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Value of the Gaussian kernel function.
        """
        dist2 = np.sum((x - y) ** 2, axis=-1, keepdims=True)
        return np.exp(-dist2 / (2 * sigma ** 2))

    def kappa_derivative(self, x, y, sigma):
        """
        Compute the derivative of the Gaussian kernel function kappa with respect to x.

        Args:
            x (np.ndarray): Coordinates of input point x.
            y (np.ndarray): Coordinates of input point y.
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Derivative of the Gaussian kernel function with respect to x.
        """
        dist2 = np.sum((x - y) ** 2, axis=-1, keepdims=True)
        kappa_value = np.exp(-dist2 / (2 * sigma ** 2))
        return -((x - y) / (sigma ** 2)) * kappa_value  
        

    def Delta_y_kappa(self, x, y, d, sigma):
        """
        Compute the Laplacian of the Gaussian kernel function kappa with respect to y.

        Args:
            x (np.ndarray): Coordinates of input point x.
            y (np.ndarray): Coordinates of input point y.
            d (int): Dimension of the input points.
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Value of the Laplacian of the Gaussian kernel function with respect to y.
        """
        dist2 = np.sum((x - y) ** 2, axis=-1, keepdims=True)
        val = (-d * (sigma ** 2) + dist2) / (sigma ** 4) * np.exp(-dist2 / (2 * sigma ** 2))
        return val

    def Delta_x_kappa(self, x, y, d, sigma):
        """
        Compute the Laplacian of the Gaussian kernel function kappa with respect to x.

        Args:
            x (np.ndarray): Coordinates of input point x.
            y (np.ndarray): Coordinates of input point y.
            d (int): Dimension of the input points.
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Value of the Laplacian of the Gaussian kernel function with respect to x.
        """
        dist2 = np.sum((x - y) ** 2, axis=-1, keepdims=True)
        val = (-d * (sigma ** 2) + dist2) / (sigma ** 4) * np.exp(-dist2 / (2 * sigma ** 2))
        return val

    def Delta_x_Delta_y_kappa(self, x, y, d, sigma):
        """
        Compute the Laplacian of the Gaussian kernel function kappa with respect to both x and y.

        Args:
            x (np.ndarray): Coordinates of input point x.
            y (np.ndarray): Coordinates of input point y.
            d (int): Dimension of the input points.
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Value of the Laplacian of the Gaussian kernel function with respect to both x and y.
        """
        dist2 = np.sum((x - y) ** 2, axis=-1, keepdims=True)
        val = ((sigma ** 4) * d * (2 + d) - 2 * (sigma ** 2) * (2 + d) * dist2 + dist2 ** 2) / (sigma ** 8) * np.exp(-dist2 / (2 * sigma ** 2))
        return val

    def get_GNkernel_train(self, x, y, wx0, wx1, wy0, wy1, d, sigma):
        """
        Compute the Gaussian-Newton kernel function for training data.

        Args:
            x (np.ndarray): Coordinates of input point x.
            y (np.ndarray): Coordinates of input point y.
            wx0 (float): Dirac coefficient for x.
            wx1 (float): Laplacian coefficient for x.
            wy0 (float): Dirac coefficient for y.
            wy1 (float): Laplacian coefficient for y.
            d (int): Dimension of the input points.
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Value of the Gaussian-Newton kernel function for training data.
        """
        return wx0 * wy0 * self.kappa(x, y, d, sigma) + wx0 * wy1 * self.Delta_y_kappa(x, y, d, sigma) + wy0 * wx1 * self.Delta_x_kappa(x, y, d, sigma) + wx1 * wy1 * self.Delta_x_Delta_y_kappa(x, y, d, sigma)

    def get_GNkernel_train_boundary(self, x, y, wy0, wy1, d, sigma):
        """
        Compute the Gaussian-Newton kernel function for boundary data.

        Args:
            x (np.ndarray): Coordinates of input point x.
            y (np.ndarray): Coordinates of input point y.
            wy0 (float): Dirac coefficient for y.
            wy1 (float): Laplacian coefficient for y.
            d (int): Dimension of the input points.
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Value of the Gaussian-Newton kernel function for boundary data.
        """
        return wy0 * self.kappa(x, y, d, sigma) + wy1 * self.Delta_y_kappa(x, y, d, sigma)

    def get_GNkernel_val_predict(self, x, y, wy0, wy1, d, sigma):
        """
        Compute the Gaussian-Newton kernel function for validation or prediction data.

        Args:
            x (np.ndarray): Coordinates of input point x.
            y (np.ndarray): Coordinates of input point y.
            wy0 (float): Dirac coefficient for y.
            wy1 (float): Laplacian coefficient for y.
            d (int): Dimension of the input points.
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Value of the Gaussian-Newton kernel function for validation or prediction data.
        """
        return wy0 * self.kappa(x, y, d, sigma) + wy1 * self.Delta_y_kappa(x, y, d, sigma)

    def assembly_Theta(self, X_domain, X_boundary, w0, w1, sigma):
        """
        Assemble the Theta matrix for the given domain and boundary points.

        Args:
            X_domain (np.ndarray): Coordinates of the domain points, shape (N_domain, d).
            X_boundary (np.ndarray): Coordinates of the boundary points, shape (N_boundary, d).
            w0 (np.ndarray): Coefficients of Diracs, shape (N_domain,).
            w1 (np.ndarray): Coefficients of Laplacians, shape (N_domain,).
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Assembled Theta matrix, shape (N_domain + N_boundary, N_domain + N_boundary).
        """
        N_domain, d = np.shape(X_domain)
        N_boundary, _ = np.shape(X_boundary)
        Theta = np.zeros((N_domain + N_boundary, N_domain + N_boundary))
        
        XdXd0 = np.reshape(np.tile(X_domain, (1, N_domain)), (-1, d))
        XdXd1 = np.tile(X_domain, (N_domain, 1))
        
        XbXd0 = np.reshape(np.tile(X_boundary, (1, N_domain)), (-1, d))
        XbXd1 = np.tile(X_domain, (N_boundary, 1))
        
        XbXb0 = np.reshape(np.tile(X_boundary, (1, N_boundary)), (-1, d))
        XbXb1 = np.tile(X_boundary, (N_boundary, 1))
        
        arr_wx0 = np.reshape(np.tile(w0, (1, N_domain)), (-1, 1))
        arr_wx1 = np.reshape(np.tile(w1, (1, N_domain)), (-1, 1))
        arr_wy0 = np.tile(w0, (N_domain, 1))
        arr_wy1 = np.tile(w1, (N_domain, 1))
        
        arr_wy0_bd = np.tile(w0, (N_boundary, 1))
        arr_wy1_bd = np.tile(w1, (N_boundary, 1))
        
        val = self.get_GNkernel_train(XdXd0, XdXd1, arr_wx0, arr_wx1, arr_wy0, arr_wy1, d, sigma)
        Theta[:N_domain, :N_domain] = np.reshape(val, (N_domain, N_domain))
        
        val = self.get_GNkernel_train_boundary(XbXd0, XbXd1, arr_wy0_bd, arr_wy1_bd, d, sigma)
        Theta[N_domain:, :N_domain] = np.reshape(val, (N_boundary, N_domain))
        Theta[:N_domain, N_domain:] = np.transpose(np.reshape(val, (N_boundary, N_domain)))
        
        val = self.kappa(XbXb0, XbXb1, d, sigma)
        Theta[N_domain:, N_domain:] = np.reshape(val, (N_boundary, N_boundary))
        
        return Theta
    
    def assembly_Theta_value_predict(self, X_infer, X_domain, X_boundary, w0, w1, sigma):
        """
        Assemble the Theta matrix for the given inference, domain, and boundary points.

        Args:
            X_infer (np.ndarray): Coordinates of the inference points, shape (N_infer, d).
            X_domain (np.ndarray): Coordinates of the domain points, shape (N_domain, d).
            X_boundary (np.ndarray): Coordinates of the boundary points, shape (N_boundary, d).
            w0 (np.ndarray): Coefficients of Diracs, shape (N_domain,).
            w1 (np.ndarray): Coefficients of Laplacians, shape (N_domain,).
            sigma (float): Parameter of the kernel function.

        Returns:
            np.ndarray: Assembled Theta matrix, shape (N_infer, N_domain + N_boundary).
        """
        N_infer, d = np.shape(X_infer)
        N_domain, _ = np.shape(X_domain)
        N_boundary, _ = np.shape(X_boundary)
        Theta = np.zeros((N_infer, N_domain + N_boundary))
        
        XiXd0 = np.reshape(np.tile(X_infer, (1, N_domain)), (-1, d))
        XiXd1 = np.tile(X_domain, (N_infer, 1))
        
        XiXb0 = np.reshape(np.tile(X_infer, (1, N_boundary)), (-1, d))
        XiXb1 = np.tile(X_boundary, (N_infer, 1))
        
        arr_wy0 = np.tile(w0, (N_infer, 1))
        arr_wy1 = np.tile(w1, (N_infer, 1))
        
        val = self.get_GNkernel_val_predict(XiXd0, XiXd1, arr_wy0, arr_wy1, d, sigma)
        Theta[:N_infer, :N_domain] = np.reshape(val, (N_infer, N_domain))
        
        val = self.kappa(XiXb0, XiXb1, d, sigma)
        Theta[:N_infer, N_domain:] = np.reshape(val, (N_infer, N_boundary))
        
        return Theta
    
    def GPsolver(self, X_domain, X_boundary, GN_step=4):
        """
        Solve the Gaussian Process for the given domain and boundary points.

        Args:
            X_domain (np.ndarray): Coordinates of the domain points, shape (N_domain, d).
            X_boundary (np.ndarray): Coordinates of the boundary points, shape (N_boundary, d).
            GN_step (int, optional): Number of Gauss-Newton iterations. Default is 4.

        Returns:
            np.ndarray: Solution after Gauss-Newton iterations, shape (N_domain,).
        """
        N_domain, d = np.shape(X_domain)
        sol = np.random.randn(N_domain,1)
        sigma = self.sigma
        nugget = self.nugget
        rhs_f = self.f(X_domain)[:, np.newaxis]
        bdy_g = self.g(X_boundary)[:, np.newaxis]
        eq= self.equation
        
        for i in range(GN_step):
            w1 = -np.ones((N_domain, 1))
            w0 = self.alpha * eq.m * (sol ** (eq.m - 1))
            Theta_train = self.assembly_Theta(X_domain, X_boundary, w0, w1, sigma)
            Theta_test = self.assembly_Theta_value_predict(X_domain, X_domain, X_boundary, w0, w1, sigma)
            rhs = rhs_f + self.alpha * (eq.m - 1) * (sol ** eq.m)
            rhs = np.concatenate((rhs, bdy_g), axis=0)
            sol = Theta_test @ (np.linalg.solve(Theta_train + nugget * np.diag(np.diag(Theta_train)), rhs))
        
        return sol
    
    def compute_gradient(self, X_domain, sol):
        """
        Compute the gradient of the solution with respect to the input data.

        Args:
            X_domain (np.ndarray): Coordinates of the domain points, shape (N_domain, d).
            sol (np.ndarray): Solution after Gauss-Newton iterations, shape (N_domain,).

        Returns:
            np.ndarray: Gradient of the solution with respect to the input data, shape (N_domain, d).
        """
        sigma = self.sigma
        N_domain, d = np.shape(X_domain)
        # Compute the derivative of the solution with respect to X_domain
        sol_derivative = np.zeros((N_domain, d))
        for i in range(N_domain):
            for j in range(N_domain):
                sol_derivative[i] += self.kappa_derivative(X_domain[i], X_domain[j], sigma) * sol[j]
        
        return sol_derivative

    # the u,f,g here in GP is very different from the u,f,g in equation.py, which is used for the MLP solver
    def u(self,x):
        return np.sin(np.sum(x,axis=-1))

    def f(self,x):
        return self.d*np.sin(np.sum(x,axis=-1))+self.alpha*(self.u(x)**self.m)

    def g(self,x):
        return self.u(x)
    

   