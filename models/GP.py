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
    
    def kappa(self, x_t, y_t, sigma):
        """
        Compute the Gaussian kernel function kappa.

        Args:
            x_t (torch.Tensor): Coordinates of input point x_t.
            y_t (torch.Tensor): Coordinates of input point y_t.
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Value of the Gaussian kernel function.
        """
        dist2 = torch.sum((x_t - y_t) ** 2, dim=-1, keepdim=True)
        return torch.exp(-dist2 / (2 * sigma ** 2))

    def kappa_derivative(self, x_t, y_t, sigma):
        """
        Compute the derivative of the Gaussian kernel function kappa with respect to x_t.

        Args:
            x_t (torch.Tensor): Coordinates of input point x_t.
            y_t (torch.Tensor): Coordinates of input point y_t.
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Derivative of the Gaussian kernel function with respect to x_t.
        """
        dist2 = torch.sum((x_t - y_t) ** 2, dim=-1, keepdim=True)
        kappa_value = torch.exp(-dist2 / (2 * sigma ** 2))
        return -((x_t - y_t) / (sigma ** 2)) * kappa_value 
        

    def Delta_y_kappa(self, x_t, y_t, sigma):
        """
        Compute the Laplacian of the Gaussian kernel function kappa with respect to y_t.

        Args:
            x_t (torch.Tensor): Coordinates of input point x_t.
            y_t (torch.Tensor): Coordinates of input point y_t.
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Value of the Laplacian of the Gaussian kernel function with respect to y_t.
        """
        dist2 = torch.sum((x_t - y_t) ** 2, dim=-1, keepdim=True)
        val = (-self.n_input * (sigma ** 2) + dist2) / (sigma ** 4) * torch.exp(-dist2 / (2 * sigma ** 2))
        return val

    def Delta_x_kappa(self, x_t, y_t, sigma):
        """
        Compute the Laplacian of the Gaussian kernel function kappa with respect to x_t.

        Args:
            x_t (torch.Tensor): Coordinates of input point x_t.
            y_t (torch.Tensor): Coordinates of input point y_t.
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Value of the Laplacian of the Gaussian kernel function with respect to x_t.
        """
        dist2 = torch.sum((x_t - y_t) ** 2, dim=-1, keepdim=True)
        val = (-self.n_input * (sigma ** 2) + dist2) / (sigma ** 4) * torch.exp(-dist2 / (2 * sigma ** 2))
        return val

    def Delta_x_Delta_y_kappa(self, x_t, y_t, sigma):
        """
        Compute the Laplacian of the Gaussian kernel function kappa with respect to both x_t and y_t.

        Args:
            x_t (torch.Tensor): Coordinates of input point x_t.
            y_t (torch.Tensor): Coordinates of input point y_t.
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Value of the Laplacian of the Gaussian kernel function with respect to both x_t and y_t.
        """
        dist2 = torch.sum((x_t - y_t) ** 2, dim=-1, keepdim=True)
        val = ((sigma ** 4) * self.n_input * (2 + self.n_input) - 2 * (sigma ** 2) * (2 + self.n_input) * dist2 + dist2 ** 2) / (sigma ** 8) * torch.exp(-dist2 / (2 * sigma ** 2))
        return val

    def get_GNkernel_train(self, x_t, y_t, wx0, wx1, wy0, wy1, sigma):
        """
        Compute the Gaussian-Newton kernel function for training data.

        Args:
            x_t (torch.Tensor): Coordinates of input point x_t.
            y_t (torch.Tensor): Coordinates of input point y_t.
            wx0 (float): Dirac coefficient for x_t.
            wx1 (float): Laplacian coefficient for x_t.
            wy0 (float): Dirac coefficient for y_t.
            wy1 (float): Laplacian coefficient for y_t.
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Value of the Gaussian-Newton kernel function for training data.
        """
        kappa_val = self.kappa(x_t, y_t, sigma)
        delta_y_kappa_val = self.Delta_y_kappa(x_t, y_t, sigma)
        delta_x_kappa_val = self.Delta_x_kappa(x_t, y_t, sigma)
        delta_x_delta_y_kappa_val = self.Delta_x_Delta_y_kappa(x_t, y_t, sigma)
        
        return (wx0 * wy0 * kappa_val + 
                wx0 * wy1 * delta_y_kappa_val + 
                wy0 * wx1 * delta_x_kappa_val + 
                wx1 * wy1 * delta_x_delta_y_kappa_val)

    def get_GNkernel_train_boundary(self, x_t, y_t, wy0, wy1, sigma):
        """
        Compute the Gaussian-Newton kernel function for boundary data.

        Args:
            x_t (torch.Tensor): Coordinates of input point x_t.
            y_t (torch.Tensor): Coordinates of input point y_t.
            wy0 (float): Dirac coefficient for y_t.
            wy1 (float): Laplacian coefficient for y_t.
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Value of the Gaussian-Newton kernel function for boundary data.
        """
        kappa_val = self.kappa(x_t, y_t, sigma)
        delta_y_kappa_val = self.Delta_y_kappa(x_t, y_t, sigma)
        
        return wy0 * kappa_val + wy1 * delta_y_kappa_val

    def get_GNkernel_val_predict(self, x_t, y_t, wy0, wy1, sigma):
        """
        Compute the Gaussian-Newton kernel function for validation or prediction data.

        Args:
            x_t (torch.Tensor): Coordinates of input point x_t.
            y_t (torch.Tensor): Coordinates of input point y_t.
            wy0 (float): Dirac coefficient for y_t.
            wy1 (float): Laplacian coefficient for y_t.
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Value of the Gaussian-Newton kernel function for validation or prediction data.
        """
        kappa_val = self.kappa(x_t, y_t, sigma)
        delta_y_kappa_val = self.Delta_y_kappa(x_t, y_t, sigma)
        
        return wy0 * kappa_val + wy1 * delta_y_kappa_val

    def assembly_Theta(self, x_t_domain, x_t_boundary, w0, w1, sigma):
        """
        Assemble the Theta matrix for the given domain and boundary points.

        Args:
            x_t_domain (torch.Tensor): Coordinates of the domain points, shape (N_domain, n_input).
            x_t_boundary (torch.Tensor): Coordinates of the boundary points, shape (N_boundary, n_input).
            w0 (torch.Tensor): Coefficients of Diracs, shape (N_domain,).
            w1 (torch.Tensor): Coefficients of Laplacians, shape (N_domain,).
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Assembled Theta matrix, shape (N_domain + N_boundary, N_domain + N_boundary).
        """
        N_domain, n_input = x_t_domain.shape
        N_boundary, _ = x_t_boundary.shape
        Theta = torch.zeros((N_domain + N_boundary, N_domain + N_boundary), dtype=torch.float32)
        
        XdXd0 = x_t_domain.repeat(1, N_domain).view(-1, n_input)
        XdXd1 = x_t_domain.repeat(N_domain, 1)
        
        XbXd0 = x_t_boundary.repeat(1, N_domain).view(-1, n_input)
        XbXd1 = x_t_domain.repeat(N_boundary, 1)
        
        XbXb0 = x_t_boundary.repeat(1, N_boundary).view(-1, n_input)
        XbXb1 = x_t_boundary.repeat(N_boundary, 1)
        
        arr_wx0 = w0.repeat(1, N_domain).view(-1, 1)
        arr_wx1 = w1.repeat(1, N_domain).view(-1, 1)
        arr_wy0 = w0.repeat(N_domain, 1)
        arr_wy1 = w1.repeat(N_domain, 1)
        
        arr_wy0_bd = w0.repeat(N_boundary, 1)
        arr_wy1_bd = w1.repeat(N_boundary, 1)
        
        val = self.get_GNkernel_train(XdXd0, XdXd1, arr_wx0, arr_wx1, arr_wy0, arr_wy1, sigma)
        Theta[:N_domain, :N_domain] = val.view(N_domain, N_domain)
        
        val = self.get_GNkernel_train_boundary(XbXd0, XbXd1, arr_wy0_bd, arr_wy1_bd, sigma)
        Theta[N_domain:, :N_domain] = val.view(N_boundary, N_domain)
        Theta[:N_domain, N_domain:] = val.view(N_boundary, N_domain).transpose(0, 1)
        
        val = self.kappa(XbXb0, XbXb1, sigma)
        Theta[N_domain:, N_domain:] = val.view(N_boundary, N_boundary)
        
        return Theta
    
    def assembly_Theta_value_predict(self, x_t_infer, x_t_domain, x_t_boundary, w0, w1, sigma):
        """
        Assemble the Theta matrix for the given inference, domain, and boundary points.

        Args:
            x_t_infer (torch.Tensor): Coordinates of the inference points, shape (N_infer, n_input).
            x_t_domain (torch.Tensor): Coordinates of the domain points, shape (N_domain, n_input).
            x_t_boundary (torch.Tensor): Coordinates of the boundary points, shape (N_boundary, n_input).
            w0 (torch.Tensor): Coefficients of Diracs, shape (N_domain,).
            w1 (torch.Tensor): Coefficients of Laplacians, shape (N_domain,).
            sigma (float): Parameter of the kernel function.

        Returns:
            torch.Tensor: Assembled Theta matrix, shape (N_infer, N_domain + N_boundary).
        """
        N_infer, n_input = x_t_infer.shape
        N_domain, _ = x_t_domain.shape
        N_boundary, _ = x_t_boundary.shape
        Theta = torch.zeros((N_infer, N_domain + N_boundary), dtype=torch.float32)
        
        XiXd0 = x_t_infer.repeat(1, N_domain).view(-1, n_input)
        XiXd1 = x_t_domain.repeat(N_infer, 1)
        
        XiXb0 = x_t_infer.repeat(1, N_boundary).view(-1, n_input)
        XiXb1 = x_t_boundary.repeat(N_infer, 1)
        
        arr_wy0 = w0.repeat(N_infer, 1)
        arr_wy1 = w1.repeat(N_infer, 1)
        
        val = self.get_GNkernel_val_predict(XiXd0, XiXd1, arr_wy0, arr_wy1, sigma)
        Theta[:N_infer, :N_domain] = val.view(N_infer, N_domain)
        
        val = self.kappa(XiXb0, XiXb1, sigma)
        Theta[:N_infer, N_domain:] = val.view(N_infer, N_boundary)
        
        return Theta
    
    def GPsolver(self, x_t_domain, x_t_boundary, GN_step=4):
        """
        Solve the Gaussian Process for the given domain and boundary points.

        Args:
            x_t_domain (np.ndarray): Coordinates of the domain points, shape (N_domain, n_input).
            x_t_boundary (np.ndarray): Coordinates of the boundary points, shape (N_boundary, n_input).
            GN_step (int, optional): Number of Gauss-Newton iterations. Default is 4.

        Returns:
            np.ndarray: Solution after Gauss-Newton iterations, shape (N_domain,).
        """
        N_domain, n_input = np.shape(x_t_domain)
        eq = self.equation
        sol = np.random.randn(N_domain,1)
        tensor_sol = torch.tensor(sol, dtype=torch.float32,requires_grad=True)
        sigma = self.sigma
        nugget = self.nugget
        bdy_g = eq.g(x_t_boundary)[:, np.newaxis]
        tensor_bdy_g= torch.tensor(bdy_g, dtype=torch.float32)
        tensor_x_t_boundary = torch.tensor(x_t_boundary, dtype=torch.float32,requires_grad=True)
        
        for i in range(GN_step):
            rhs_f = eq.f(x_t_domain,sol,self.compute_gradient(x_t_domain,sol)[:,:-1])
            tensor_rhs_f = torch.tensor(rhs_f, dtype=torch.float32)
            tensor_w1 = -torch.ones((N_domain, 1), dtype=torch.float32)
            if i == 0:
                tensor_PDE_grad = torch.randn(N_domain,1,dtype=torch.float32)
            else:
                g_loss = eq.gPDE_loss(tensor_x_t_domain, tensor_sol)[:-1] 
                '''cuda out of memory error occurs when using the line above'''
                tensor_PDE_grad = torch.sum(torch.cat(g_loss,dim=1),dim=1,keepdim=True)
            tensor_w0 = self.alpha * tensor_PDE_grad
            tensor_x_t_domain = torch.tensor(x_t_domain, dtype=torch.float32,requires_grad=True)
            Theta_train = self.assembly_Theta(tensor_x_t_domain, tensor_x_t_boundary, tensor_w0, tensor_w1, sigma)
            Theta_test = self.assembly_Theta_value_predict(tensor_x_t_domain, tensor_x_t_domain, tensor_x_t_boundary, tensor_w0, tensor_w1, sigma)
            tensor_rhs= tensor_rhs_f + self.alpha * tensor_PDE_grad * tensor_sol
            tensor_rhs = torch.cat((tensor_rhs, tensor_bdy_g), dim=0)
            tensor_sol =torch.matmul(Theta_test,(torch.linalg.solve(Theta_train + nugget * torch.diag(torch.diag(Theta_train)), tensor_rhs)))
            sol = tensor_sol.cpu().detach().numpy()
        self.sol = sol
        self.x_t_domain = x_t_domain
        self.x_t_boundary = x_t_boundary
        self.right_op = torch.linalg.solve(Theta_train + nugget * torch.diag(torch.diag(Theta_train)), tensor_rhs).cpu().detach().numpy()
        self.w0 = tensor_w0.cpu().detach().numpy()
        self.w1 = tensor_w1.cpu().detach().numpy()
        return sol
    
    def predict(self, x_t_infer):
        """
        Predict the solution at the given inference points.

        Args:
            x_t_infer (np.ndarray): Coordinates of the inference points, shape (N_infer, n_input).

        Returns:
            np.ndarray: Predicted solution at the inference points, shape (N_infer,).
        """
        sigma = self.sigma
        tensor_x_t_infer = torch.tensor(x_t_infer, dtype=torch.float32)
        tensor_x_t_domain = torch.tensor(self.x_t_domain, dtype=torch.float32)
        tensor_x_t_boundary = torch.tensor(self.x_t_boundary, dtype=torch.float32)
        w1 = self.w1
        tensor_w1 = torch.tensor(w1, dtype=torch.float32)
        w0 = self.w0
        tensor_w0 = torch.tensor(w0, dtype=torch.float32)
        Theta_test = self.assembly_Theta_value_predict(tensor_x_t_infer, tensor_x_t_domain, tensor_x_t_boundary, tensor_w0, tensor_w1, sigma)
        tensor_right_op = torch.tensor(self.right_op,dtype=torch.float32)
        new_sol = (torch.matmul(Theta_test,tensor_right_op)).cpu().detach().numpy()
        
        return new_sol
    
    def compute_gradient(self, x_t_infer, sol):
        """
        Compute the gradient of the solution with respect to the input data.

        Args:
            x_t_infer (np.ndarray): Coordinates of the infer points, shape (N_infer, n_input).
            sol (np.ndarray): Solution after Gauss-Newton iterations, shape (N_infer,).

        Returns:
            np.ndarray: Gradient of the solution with respect to the input data, shape (N_infer, n_input).
        """
        sigma = self.sigma
        tensor_sol= torch.tensor(sol, dtype=torch.float32)
        N_infer, n_input = np.shape(x_t_infer)
        tensor_x_t_infer = torch.tensor(x_t_infer, dtype=torch.float32)
        # Compute the derivative of the solution with respect to x_t_infer
        tensor_sol_derivative = torch.zeros((N_infer, n_input), dtype=torch.float32)
        for i in range(N_infer):
            for j in range(N_infer):
                tensor_sol_derivative[i] += self.kappa_derivative(tensor_x_t_infer[i], tensor_x_t_infer[j], sigma) * tensor_sol[j]
        sol_derivative = tensor_sol_derivative.cpu().detach().numpy()
        return sol_derivative

    

   