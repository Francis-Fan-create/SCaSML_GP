import numpy as np
import torch
class GP(object):
    '''Gaussian Kernel Solver for high dimensional PDE'''
    def __init__(self,equation):
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
        self.sigma= equation.sigma() # Standard deviation 
        self.nugget = 1e-10  # Regularization parameter to ensure numerical stability
    
    def kappa(self,x_t,y_t):
        '''Compute the kernel matrix K(x_t,y_t)'''
        return np.exp(-np.sum((x_t-y_t)**2)/(2*self.sigma**2))  # Gaussian kernel
    
    def dx_t_kappa(self,x_t,y_t):
        '''Compute gradient of the kernel matrix K(x_t,y_t) with respect to x_t'''
        return -(x_t-y_t)/self.sigma**2*self.kappa(x_t,y_t)
    
    def dt_x_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the kernel matrix K(x_t,y_t) with respect to x_t'''
        time_x_t= x_t[-1]
        time_y_t= y_t[-1]
        return -(time_x_t-time_y_t)/self.sigma**2*self.kappa(x_t,y_t)
    
    def dt_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the kernel matrix K(x_t,y_t) with respect to y_t'''
        return -self.dt_x_t_kappa(x_t,y_t)
    
    def div_x_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to x_t'''
        x= x_t[:-1]
        y= y_t[:-1]
        return -np.sum(x-y)/self.sigma**2*self.kappa(x_t,y_t)
    
    def div_y_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to y_t'''
        return -self.div_x_kappa(x_t,y_t)
    
    def laplacian_x_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t'''
        x= x_t[:-1]
        y= y_t[:-1]
        dist2= np.sum((x-y)**2)
        return (-self.d/self.sigma**2+dist2/self.sigma**4)*self.kappa(x_t,y_t)
    
    def laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the kernel matrix K(x_t,y_t) with respect to y_t'''
        return self.laplacian_x_t_kappa(x_t,y_t)
    
    def dt_x_t_dt_y_t_kappa(self,x_t,y_t):
        '''Compute second time derivative of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        time_x_t = x_t[-1]
        time_y_t = y_t[-1]
        return (1/self.sigma**2-(time_x_t-time_y_t)**2/self.sigma**4)*self.kappa(x_t,y_t)   
    
    def dt_x_t_div_y_kappa(self,x_t,y_t):
        '''Compute time derivative of the divergence of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        time_x_t= x_t[-1]
        time_y_t= y_t[-1]
        return (time_y_t-time_x_t)/self.sigma**2*self.div_y_kappa(x_t,y_t)
    
    def dt_x_t_laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        time_x_t= x_t[-1]
        time_y_t= y_t[-1]
        return (time_y_t-time_x_t)/self.sigma**2*self.laplacian_y_t_kappa(x_t,y_t)
    
    def div_x_dt_y_t_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to x_t'''
        x= x_t[:-1]
        y= y_t[:-1]
        time_x_t= x_t[-1]
        time_y_t= y_t[-1]
        return -(time_x_t-time_y_t)*np.sum(x-y)/self.sigma**2*self.kappa(x_t,y_t)
    
    def div_x_div_y_kappa(self,x_t,y_t):
        '''Compute divergence of the divergence of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        x= x_t[:-1]
        y= y_t[:-1]
        div = np.sum(x-y)
        return (self.d/self.sigma**2-div**2/self.sigma**4)*self.kappa(x_t,y_t)
    
    def div_x_laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute divergence of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        x= x_t[:-1]
        y= y_t[:-1]
        div = np.sum(x-y)/self.sigma**2
        kappa= self.kappa(x_t,y_t)
        term1 = (self.d+1)*div/self.sigma**2*kappa
        term2 = div/self.sigma**4*kappa
        dist2= np.sum((x-y)**2)
        term3 = -dist2*div/self.sigma**6*kappa
        return term1+term2+term3
    
    def laplacian_x_t_dt_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        time_x_t= x_t[-1]
        time_y_t= y_t[-1]
        return (time_x_t-time_y_t)/self.sigma**2*self.laplacian_x_t_kappa(x_t,y_t)
    
    def laplacian_x_t_div_y_kappa(self,x_t,y_t):
        '''Compute Laplacian of the divergence of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        x= x_t[:-1]
        y= y_t[:-1]
        div = np.sum(y-x)/self.sigma**2
        kappa= self.kappa(x_t,y_t)
        term1 = (self.d+1)*div/self.sigma**2*kappa
        term2 = div/self.sigma**4*kappa
        dist2= np.sum((y-x)**2)
        term3 = -dist2*div/self.sigma**6*kappa
        return term1+term2+term3
    
    def laplacian_x_t_laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        x= x_t[:-1]
        y= y_t[:-1]
        dist2= np.sum((x-y)**2)
        d = self.d
        kappa= self.kappa(x_t,y_t)
        sigma = self.sigma
        return (d*(d+2)/sigma**4-2*(d+2)*dist2/sigma**6+dist2**2/sigma**8)*kappa
    
    def kernel_phi_phi(self, x_t_domain, x_t_boundary):
        '''Compute the kernel matrix K(phi,phi)'''
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        phi_dim = N_domain*4+N_boundary
        kernel_phi_phi = np.zeros((phi_dim, phi_dim))
        for i in range(N_domain):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.kappa(x_t_domain[i], x_t_domain[j])
            for j in range(N_domain, 2*N_domain):
                kernel_phi_phi[i, j] = self.dt_y_t_kappa(x_t_domain[i], x_t_domain[j-N_domain])
            for j in range(2*N_domain, 3*N_domain):
                kernel_phi_phi[i, j] = self.div_y_kappa(x_t_domain[i], x_t_domain[j-2*N_domain])
            for j in range(3*N_domain, 4*N_domain):
                kernel_phi_phi[i, j] = self.laplacian_y_t_kappa(x_t_domain[i], x_t_domain[j-3*N_domain])
            for j in range(4*N_domain, 4*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.kappa(x_t_domain[i], x_t_boundary[j-4*N_domain])
        for i in range(N_domain, 2*N_domain):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.dt_x_t_kappa(x_t_domain[i-N_domain], x_t_domain[j])
            for j in range(N_domain, 2*N_domain):
                kernel_phi_phi[i, j] = self.dt_x_t_dt_y_t_kappa(x_t_domain[i-N_domain], x_t_domain[j-N_domain])
            for j in range(2*N_domain, 3*N_domain):
                kernel_phi_phi[i, j] = self.dt_x_t_div_y_kappa(x_t_domain[i-N_domain], x_t_domain[j-2*N_domain])
            for j in range(3*N_domain, 4*N_domain):
                kernel_phi_phi[i, j] = self.dt_x_t_laplacian_y_t_kappa(x_t_domain[i-N_domain], x_t_domain[j-3*N_domain])
            for j in range(4*N_domain, 4*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.dt_x_t_kappa(x_t_domain[i-N_domain], x_t_boundary[j-4*N_domain])
        for i in range(2*N_domain, 3*N_domain):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.div_x_kappa(x_t_domain[i-2*N_domain], x_t_domain[j])
            for j in range(N_domain, 2*N_domain):
                kernel_phi_phi[i, j] = self.div_x_dt_y_t_kappa(x_t_domain[i-2*N_domain], x_t_domain[j-N_domain])
            for j in range(2*N_domain, 3*N_domain):
                kernel_phi_phi[i, j] = self.div_x_div_y_kappa(x_t_domain[i-2*N_domain], x_t_domain[j-2*N_domain])
            for j in range(3*N_domain, 4*N_domain):
                kernel_phi_phi[i, j] = self.div_x_laplacian_y_t_kappa(x_t_domain[i-2*N_domain], x_t_domain[j-3*N_domain])
            for j in range(4*N_domain, 4*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.div_x_kappa(x_t_domain[i-2*N_domain], x_t_boundary[j-4*N_domain])
        for i in range(3*N_domain, 4*N_domain):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.laplacian_x_t_kappa(x_t_domain[i-3*N_domain], x_t_domain[j])
            for j in range(N_domain, 2*N_domain):
                kernel_phi_phi[i, j] = self.laplacian_x_t_dt_y_t_kappa(x_t_domain[i-3*N_domain], x_t_domain[j-N_domain])
            for j in range(2*N_domain, 3*N_domain):
                kernel_phi_phi[i, j] = self.laplacian_x_t_div_y_kappa(x_t_domain[i-3*N_domain], x_t_domain[j-2*N_domain])
            for j in range(3*N_domain, 4*N_domain):
                kernel_phi_phi[i, j] = self.laplacian_x_t_laplacian_y_t_kappa(x_t_domain[i-3*N_domain], x_t_domain[j-3*N_domain])
            for j in range(4*N_domain, 4*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.laplacian_x_t_kappa(x_t_domain[i-3*N_domain], x_t_boundary[j-4*N_domain])
        for i in range(4*N_domain, 4*N_domain+N_boundary):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.kappa(x_t_boundary[i-4*N_domain], x_t_domain[j])
            for j in range(N_domain, 2*N_domain):
                kernel_phi_phi[i, j] = self.dt_y_t_kappa(x_t_boundary[i-4*N_domain], x_t_domain[j-N_domain])
            for j in range(2*N_domain, 3*N_domain):
                kernel_phi_phi[i, j] = self.div_y_kappa(x_t_boundary[i-4*N_domain], x_t_domain[j-2*N_domain])
            for j in range(3*N_domain, 4*N_domain):
                kernel_phi_phi[i, j] = self.laplacian_y_t_kappa(x_t_boundary[i-4*N_domain], x_t_domain[j-3*N_domain])
            for j in range(4*N_domain, 4*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.kappa(x_t_boundary[i-4*N_domain], x_t_boundary[j-4*N_domain])
        return kernel_phi_phi
    
    def kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the kernel matrix K(x_t,phi)'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        row_dim = N_infer
        col_dim = N_domain*4+N_boundary
        kernel_x_phi = np.zeros((row_dim, col_dim))
        for i in range(N_infer):
            for j in range(N_domain):
                kernel_x_phi[i, j] = self.kappa(x_t_infer[i], x_t_domain[j])
            for j in range(N_domain, 2*N_domain):
                kernel_x_phi[i, j] = self.dt_y_t_kappa(x_t_infer[i], x_t_domain[j-N_domain])
            for j in range(2*N_domain, 3*N_domain):
                kernel_x_phi[i, j] = self.div_y_kappa(x_t_infer[i], x_t_domain[j-2*N_domain])
            for j in range(3*N_domain, 4*N_domain):
                kernel_x_phi[i, j] = self.laplacian_y_t_kappa(x_t_infer[i], x_t_domain[j-3*N_domain])
            for j in range(4*N_domain, 4*N_domain+N_boundary):
                kernel_x_phi[i, j] = self.kappa(x_t_infer[i], x_t_boundary[j-4*N_domain])
        return kernel_x_phi
    
    def dx_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the gradient of the kernel matrix K(x_t,phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        row_dim = N_infer
        col_dim = N_domain*4+N_boundary
        dx_t_kernel_x_phi = np.zeros((row_dim, col_dim, self.n_input))
        for i in range(N_infer):
            for j in range(N_domain):
                dx_t_kernel_x_phi[i, j] = self.kappa(x_t_infer[i], x_t_domain[j])*(x_t_infer[i]-x_t_domain[j])/self.sigma**2
            for j in range(N_domain, 2*N_domain):
                dx_t_kernel_x_phi[i, j] = self.dt_y_t_kappa(x_t_infer[i], x_t_domain[j-N_domain])*(x_t_infer[i]-x_t_domain[j-N_domain])/self.sigma**2
            for j in range(2*N_domain, 3*N_domain):
                dx_t_kernel_x_phi[i, j] = self.div_y_kappa(x_t_infer[i], x_t_domain[j-2*N_domain])*(x_t_infer[i]-x_t_domain[j-2*N_domain])/self.sigma**2
            for j in range(3*N_domain, 4*N_domain):
                dx_t_kernel_x_phi[i, j] = self.laplacian_y_t_kappa(x_t_infer[i], x_t_domain[j-3*N_domain])*(x_t_infer[i]-x_t_domain[j-3*N_domain])/self.sigma**2
            for j in range(4*N_domain, 4*N_domain+N_boundary):
                dx_t_kernel_x_phi[i, j] = self.kappa(x_t_infer[i], x_t_boundary[j-4*N_domain])*(x_t_infer[i]-x_t_boundary[j-4*N_domain])/self.sigma**2
        return dx_t_kernel_x_phi
    
    def f(self,x_t):
        '''Compute the nonlinear term on the right at x_t'''
        raise NotImplementedError
    
    def g(self,x_t):
        '''Compute the terminal condition at x_t'''
        return self.equation.g(x_t)[:,np.newaxis]
    
    def P(self,z):
        '''Compute the value of operator P at z'''
        return NotImplementedError
    
    def dP(self,z):
        '''Compute the value of the derivative of operator P at z'''
        return NotImplementedError
    
    def GPsolver(self, x_t_domain, x_t_boundary, GN_step=4):
        """
        Solve the Gaussian Process for the given domain and boundary points.

        Args:
            x_t_domain (np.ndarray): Coordinates of the domain points, shape (N_domain, n_input).
            x_t_boundary (np.ndarray): Coordinates of the boundary points, shape (N_boundary, n_input).
            GN_step (int, optional): Number of Gauss-Newton iterations. Default is 4.

        Returns:
            np.ndarray: Solution after Gauss-Newton iterations on x_t_domain, shape (N_domain,1).
        """
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        self.x_t_domain = x_t_domain
        self.x_t_boundary = x_t_boundary
        z = np.random.randn(4*N_domain+N_boundary, 1)
        sol_domain = np.random.randn(N_domain,1)
        sol_boundary = np.random.randn(N_boundary,1)
        sol = np.concatenate((sol_domain, sol_boundary), axis=0) 
        kernel_phi_phi = self.kernel_phi_phi(x_t_domain, x_t_boundary)  # Kernel matrix between z and z
        kernel_x_phi = self.kernel_x_t_phi(np.concatenate((x_t_domain,x_t_boundary),axis=0), x_t_domain, x_t_boundary) # Kernel matrix between x_t_domain and z
        # Compute the Moore-Penrose pseudoinverse of kernel_phi_phi
        U, S, Vt = np.linalg.svd(kernel_phi_phi)
        S_inv = np.diag(1 / S)
        kernel_phi_phi_inv = Vt.T @ S_inv @ U.T
        for i in range(GN_step):
            z[:4*N_domain] = np.repeat(self.f(x_t_domain),4,axis=0).reshape(-1,1) -self.P(z)  + np.repeat(self.dP(z)*sol_domain,4,axis=0).reshape(-1,1)
            z[4*N_domain:] = self.g(x_t_boundary) - z[4*N_domain:] + z[4*N_domain:]*sol_boundary
            right_vector = kernel_phi_phi_inv @ z
            sol = kernel_x_phi @ right_vector
            sol_domain = sol[:N_domain]
            sol_boundary = sol[N_domain:]
        self.right_vector = right_vector
        return sol_domain
    
    def predict(self,x_t_infer):
        '''Predict the solution at x_t_infer'''
        kernel_x_phi = self.kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary) # Kernel matrix between x_t_domain and z
        sol_infer = kernel_x_phi @ self.right_vector
        return sol_infer
    
    def compute_gradient(self,x_t_infer,sol):
        '''Compute the gradient of the solution at x_t_infer'''
        N_infer = x_t_infer.shape[0]
        dx_t_kernel_x_t_phi = self.dx_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)
        gradient = np.zeros((N_infer,self.n_input))
        for i in range(N_infer):
            gradient[i] = (dx_t_kernel_x_t_phi[i].T @ self.right_vector)[:,0]
        return gradient
    
class GP_Complicated_HJB(GP):
    '''Gaussian Kernel Solver for Complicated HJB'''
    def __init__(self,equation):
        super(GP_Complicated_HJB, self).__init__(equation)
    
    def f(self, x_t):
        '''Compute the nonlinear term on the right at x_t'''
        N_domain = self.x_t_domain.shape[0]
        return -2*np.ones((N_domain,1))
    
    def P(self, z):
        '''Compute the value of operator P at z'''
        N_domain = self.x_t_domain.shape[0]
        d= self.d
        P = np.repeat(z[N_domain:2*N_domain]-(1/d)*z[2*N_domain:3*N_domain]+z[3*N_domain:4*N_domain],4,axis=0).reshape(-1,1)
        return P
    
    def dP(self, z):
        '''Compute the value of the derivative of operator P at z'''
        N_domain = self.x_t_domain.shape[0]
        d= self.d
        dP = z[N_domain:2*N_domain]-(1/d)*z[2*N_domain:3*N_domain]+z[3*N_domain:4*N_domain]
        return dP
    
class GP_Explicit_Solution_Example(GP):
    '''Gaussian Kernel Solver for the Explicit Solution Example'''
    def __init__(self,equation):
        super(GP_Explicit_Solution_Example, self).__init__(equation)
    
    def f(self, x_t):
        '''Compute the nonlinear term on the right at x_t'''
        tensor_x_t = torch.tensor(x_t, dtype=torch.float32, requires_grad=True)
        tensor_t = tensor_x_t[:, -1]
        tensor_x = tensor_x_t[:, :-1]
        sum_x = torch.sum(tensor_x, axis=1)
        tensor_result=sum_x+(self.T-tensor_t)
        tensor_result=tensor_result.unsqueeze(1)
        tensor_div_x =torch.sum(torch.autograd.grad(tensor_result, tensor_x, grad_outputs=torch.ones_like(tensor_result), create_graph=True)[0],dim=1,keepdim=True)
        tensor_f = -self.sigma**2*tensor_result*tensor_div_x
        f = tensor_f.detach().cpu().numpy()
        return f
    
    def P(self, z):
        '''Compute the value of operator P at z'''
        N_domain = self.x_t_domain.shape[0]
        d= self.d
        P = np.repeat(z[N_domain:2*N_domain]-(1/d+self.sigma**2/2)*z[2*N_domain:3*N_domain]+z[3*N_domain:4*N_domain],4,axis=0).reshape(-1,1)
        return P
    
    def dP(self, z):
        '''Compute the value of the derivative of operator P at z'''
        N_domain = self.x_t_domain.shape[0]
        d = self.d
        dP = z[N_domain:2*N_domain]-(1/d+self.sigma**2/2)*z[2*N_domain:3*N_domain]+z[3*N_domain:4*N_domain]
        return dP
    

    
    
