import numpy as np
import torch
from scipy.linalg import solve_triangular,pinv
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma



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
        self.sigma= equation.sigma()*np.sqrt(self.d) # Standard deviation for Gaussian kernel
        self.nugget = 1e-5  # Regularization parameter to ensure numerical stability
    
    def kappa(self,x_t,y_t):
        '''Compute the kernel entry K(x_t,y_t) for single vector x_t and y_t'''
        return np.exp(-np.sum((x_t-y_t)**2)/(2*self.sigma**2))  # Gaussian kernel
    
    def kappa_kernel(self,x_t,y_t):
        '''Compute the kernel matrix K(x_t,y_t) for batched vectors x_t and y_t'''
        N_x = x_t.shape[0]
        N_y = y_t.shape[0]
        kernel = np.zeros((N_x,N_y))
        for i in range(N_x):
            for j in range(N_y):
                kernel[i,j] = self.kappa(x_t[i],y_t[j])
        return kernel
    
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
        return -(time_x_t-time_y_t)**2/self.sigma**4*self.kappa(x_t,y_t)
    
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
        phi_dim = 4* N_domain+N_boundary
        kernel_phi_phi = np.zeros((phi_dim, phi_dim))
        for i in range(N_domain):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.kappa(x_t_domain[i], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.kappa(x_t_domain[i], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.laplacian_y_t_kappa(x_t_domain[i], x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.dt_y_t_kappa(x_t_domain[i], x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, phi_dim):
                kernel_phi_phi[i, j] = self.div_y_kappa(x_t_domain[i], x_t_domain[j-3*N_domain-N_boundary])
        for i in range(N_domain, N_domain+N_boundary):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.kappa(x_t_boundary[i-N_domain], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.kappa(x_t_boundary[i-N_domain], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.laplacian_y_t_kappa(x_t_boundary[i-N_domain], x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.dt_y_t_kappa(x_t_boundary[i-N_domain], x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, phi_dim):
                kernel_phi_phi[i, j] = self.div_y_kappa(x_t_boundary[i-N_domain], x_t_domain[j-3*N_domain-N_boundary])
        for i in range(N_domain+N_boundary, 2*N_domain+N_boundary):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.laplacian_x_t_kappa(x_t_domain[i-N_domain-N_boundary], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.laplacian_x_t_kappa(x_t_domain[i-N_domain-N_boundary], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.laplacian_x_t_laplacian_y_t_kappa(x_t_domain[i-N_domain-N_boundary], x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.laplacian_x_t_dt_y_t_kappa(x_t_domain[i-N_domain-N_boundary], x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, phi_dim):
                kernel_phi_phi[i, j] = self.laplacian_x_t_div_y_kappa(x_t_domain[i-N_domain-N_boundary], x_t_domain[j-3*N_domain-N_boundary])
        for i in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.dt_x_t_kappa(x_t_domain[i-2*N_domain-N_boundary], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.dt_x_t_kappa(x_t_domain[i-2*N_domain-N_boundary], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.dt_x_t_laplacian_y_t_kappa(x_t_domain[i-2*N_domain-N_boundary], x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.dt_x_t_dt_y_t_kappa(x_t_domain[i-2*N_domain-N_boundary], x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, phi_dim):
                kernel_phi_phi[i, j] = self.dt_x_t_div_y_kappa(x_t_domain[i-2*N_domain-N_boundary], x_t_domain[j-3*N_domain-N_boundary])
        for i in range(3*N_domain+N_boundary, phi_dim):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.div_x_kappa(x_t_domain[i-3*N_domain-N_boundary], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.div_x_kappa(x_t_domain[i-3*N_domain-N_boundary], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.div_x_laplacian_y_t_kappa(x_t_domain[i-3*N_domain-N_boundary], x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.div_x_dt_y_t_kappa(x_t_domain[i-3*N_domain-N_boundary], x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, phi_dim):
                kernel_phi_phi[i, j] = self.div_x_div_y_kappa(x_t_domain[i-3*N_domain-N_boundary], x_t_domain[j-3*N_domain-N_boundary])
        return kernel_phi_phi
    
    def kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the kernel matrix K(x_t,phi)'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        row_dim = N_infer
        col_dim =4 * N_domain+N_boundary
        '''Compute the kernel matrix K(x_t,phi)'''
        kernel_x_phi = np.zeros((row_dim, col_dim))
        for i in range(N_infer):
            for j in range(N_domain):
                kernel_x_phi[i, j] = self.kappa(x_t_infer[i], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_x_phi[i, j] = self.kappa(x_t_infer[i], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                kernel_x_phi[i, j] = self.laplacian_y_t_kappa(x_t_infer[i], x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                kernel_x_phi[i, j] = self.dt_y_t_kappa(x_t_infer[i], x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, col_dim):
                kernel_x_phi[i, j] = self.div_y_kappa(x_t_infer[i], x_t_domain[j-3*N_domain-N_boundary])
        return kernel_x_phi
    
    def dx_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the gradient of the kernel matrix K(x_t,phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        row_dim = N_infer
        col_dim = 4* N_domain+N_boundary
        dx_t_kernel_x_phi = np.zeros((row_dim, col_dim, self.n_input))
        for i in range(N_infer):
            for j in range(N_domain):
                dx_t_kernel_x_phi[i, j] = self.dx_t_kappa(x_t_infer[i], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                dx_t_kernel_x_phi[i, j] = self.dx_t_kappa(x_t_infer[i], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                dx_t_kernel_x_phi[i, j] = self.laplacian_y_t_kappa(x_t_infer[i], x_t_domain[j-N_domain-N_boundary])*-1/self.sigma**2*(x_t_infer[i]-x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                dx_t_kernel_x_phi[i, j] = self.dt_y_t_kappa(x_t_infer[i], x_t_domain[j-2*N_domain-N_boundary])*-1/self.sigma**2*(x_t_infer[i]-x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, col_dim):
                dx_t_kernel_x_phi[i, j] = self.div_y_kappa(x_t_infer[i], x_t_domain[j-3*N_domain-N_boundary])*-1/self.sigma**2*(x_t_infer[i]-x_t_domain[j-3*N_domain-N_boundary])
        return dx_t_kernel_x_phi
    
    def laplacian_x_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the Laplacian of the kernel matrix K(x_t,phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        row_dim = N_infer
        col_dim = 4* N_domain+N_boundary
        laplacian_x_t_kernel_x_phi = np.zeros((row_dim, col_dim))
        for i in range(N_infer):
            for j in range(N_domain):
                laplacian_x_t_kernel_x_phi[i, j] = self.laplacian_x_t_kappa(x_t_infer[i], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                laplacian_x_t_kernel_x_phi[i, j] = self.laplacian_x_t_kappa(x_t_infer[i], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                laplacian_x_t_kernel_x_phi[i, j] = self.laplacian_x_t_laplacian_y_t_kappa(x_t_infer[i], x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                laplacian_x_t_kernel_x_phi[i, j] = self.laplacian_x_t_dt_y_t_kappa(x_t_infer[i], x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, col_dim):
                laplacian_x_t_kernel_x_phi[i, j] = self.laplacian_x_t_div_y_kappa(x_t_infer[i], x_t_domain[j-3*N_domain-N_boundary])
        return laplacian_x_t_kernel_x_phi
    
    def dt_x_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the time derivative of the kernel matrix K(x_t,phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        row_dim = N_infer
        col_dim = 4* N_domain+N_boundary
        dt_x_t_kernel_x_phi = np.zeros((row_dim, col_dim))
        for i in range(N_infer):
            for j in range(N_domain):
                dt_x_t_kernel_x_phi[i, j] = self.dt_x_t_kappa(x_t_infer[i], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                dt_x_t_kernel_x_phi[i, j] = self.dt_x_t_kappa(x_t_infer[i], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                dt_x_t_kernel_x_phi[i, j] = self.dt_x_t_laplacian_y_t_kappa(x_t_infer[i], x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                dt_x_t_kernel_x_phi[i, j] = self.dt_x_t_dt_y_t_kappa(x_t_infer[i], x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, col_dim):
                dt_x_t_kernel_x_phi[i, j] = self.dt_x_t_div_y_kappa(x_t_infer[i], x_t_domain[j-3*N_domain-N_boundary])
        return dt_x_t_kernel_x_phi
    
    def div_x_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the divergence of the kernel matrix K(x_t,phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        row_dim = N_infer
        col_dim = 4* N_domain+N_boundary
        div_x_t_kernel_x_phi = np.zeros((row_dim, col_dim))
        for i in range(N_infer):
            for j in range(N_domain):
                div_x_t_kernel_x_phi[i, j] = self.div_x_kappa(x_t_infer[i], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                div_x_t_kernel_x_phi[i, j] = self.div_x_kappa(x_t_infer[i], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, 2*N_domain+N_boundary):
                div_x_t_kernel_x_phi[i, j] = self.div_x_laplacian_y_t_kappa(x_t_infer[i], x_t_domain[j-N_domain-N_boundary])
            for j in range(2*N_domain+N_boundary, 3*N_domain+N_boundary):
                div_x_t_kernel_x_phi[i, j] = self.div_x_dt_y_t_kappa(x_t_infer[i], x_t_domain[j-2*N_domain-N_boundary])
            for j in range(3*N_domain+N_boundary, col_dim):
                div_x_t_kernel_x_phi[i, j] = self.div_x_div_y_kappa(x_t_infer[i], x_t_domain[j-3*N_domain-N_boundary])
        return div_x_t_kernel_x_phi
    
    
    def y_domain(self,x_t):
        '''Compute the nonlinear term on the right at x_t'''
        raise NotImplementedError
    
    def y_boundary(self,x_t):
        '''Compute the terminal condition at x_t'''
        return self.equation.g(x_t)[:,np.newaxis]
    

    def GPsolver(self, x_t_domain, x_t_boundary, GN_step=4):
        '''Solve the Gaussian Process for the given domain and boundary points'''
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        self.x_t_domain = x_t_domain
        self.x_t_boundary = x_t_boundary
        self.N_domain = N_domain
        self.N_boundary = N_boundary
        z_k = np.random.uniform(-1,1,(4* N_domain+N_boundary,1))
        y = np.concatenate((self.y_domain(x_t_domain),self.y_boundary(x_t_boundary)), axis=0)
        gamma = np.zeros((N_domain+N_boundary,1))
        kernel_phi_phi = self.kernel_phi_phi(x_t_domain, x_t_boundary) # Kernel matrix between z and z
        for i in range(GN_step):
            DF_k = self.DF(z_k)
            proj_kernel_phi_phi = DF_k @ kernel_phi_phi @ DF_k.T
            gamma = np.linalg.solve(proj_kernel_phi_phi+self.nugget*np.eye(N_domain+N_boundary), (y-self.F(z_k)+DF_k @ z_k))
            z_k_plus_1 = kernel_phi_phi @ DF_k.T @ gamma
            z_k = z_k_plus_1
        self.right_vector = np.linalg.solve(kernel_phi_phi+self.nugget*np.eye(4* N_domain+N_boundary), z_k)
        return z_k[:N_domain]
    
    def predict(self,x_t_infer):
        '''Predict the solution at x_t_infer'''
        kernel_x_t_phi = self.kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary) # Kernel matrix between x_t_domain and z
        right_vector = self.right_vector
        sol_infer = kernel_x_t_phi @ right_vector
        return sol_infer
    
    def compute_gradient(self,x_t_infer,sol):
        '''Compute the gradient of the solution at x_t_infer'''
        N_infer = x_t_infer.shape[0]
        dx_t_kernel_x_t_phi = self.dx_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)
        gradient = np.zeros((N_infer,self.n_input))
        for i in range(N_infer):
            gradient[i] = (dx_t_kernel_x_t_phi[i].T @ self.right_vector)[:,0]
        return gradient
    
    def compute_PDE_loss(self,x_t_infer):
        '''Compute the PDE loss at x_t_infer'''
        raise NotImplementedError
    
class GP_Complicated_HJB(GP):
    '''Gaussian Kernel Solver for Complicated HJB'''
    def __init__(self,equation):
        super(GP_Complicated_HJB, self).__init__(equation)
    
    def y_domain(self,x_t):
        '''Compute the nonlinear term on the right at x_t'''
        return -2*np.ones((x_t.shape[0],1))

    def F(self,z):
        '''Compute the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        d = self.d
        z_2 = z[N_domain:N_domain+N_boundary]
        z_3 = z[N_domain+N_boundary:2*N_domain+N_boundary]
        z_4 = z[2*N_domain+N_boundary:3*N_domain+N_boundary]
        z_5 = z[3*N_domain+N_boundary:]
        F = np.zeros((N_domain+N_boundary,1))
        F[:N_domain] = z_4-(1/d)*z_5+z_3
        F[N_domain:] = z_2
        return F
    
    def DF(self,z):
        '''Compute the jacobian of the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        DF = np.zeros((N_domain+N_boundary,4*N_domain+N_boundary))
        d = self.d
        for i in range(N_domain):
            DF[i,:N_domain] = 0
            DF[i,N_domain:N_domain+N_boundary] = 0
            DF[i,N_domain+N_boundary:2*N_domain+N_boundary] = 1
            DF[i,2*N_domain+N_boundary:3*N_domain+N_boundary] = 1
            DF[i,3*N_domain+N_boundary:] = -1/d
        for i in range(N_domain,N_domain+N_boundary):
            DF[i,:N_domain] = 0
            DF[i,N_domain:N_domain+N_boundary] = 1
            DF[i,N_domain+N_boundary:2*N_domain+N_boundary] = 0
            DF[i,2*N_domain+N_boundary:3*N_domain+N_boundary] = 0
            DF[i,3*N_domain+N_boundary:] = 0
        return DF
    
    def compute_PDE_loss(self, x_t_infer):
        '''Compute the PDE loss at x_t_infer'''
        d = self.d
        right_vector = self.right_vector
        dt_x_t_sol_kernel = self.dt_x_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)
        div_x_sol_kernel = self.div_x_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)
        laplacian_x_t_sol_kernel = self.laplacian_x_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)
        dt_x_t_sol = dt_x_t_sol_kernel @ right_vector
        div_x_sol = div_x_sol_kernel @ right_vector
        laplacian_x_t_sol = laplacian_x_t_sol_kernel @ right_vector
        loss = dt_x_t_sol-(1/d)*div_x_sol+laplacian_x_t_sol+2
        return loss

class GP_Explicit_Solution_Example(GP):
    '''Gaussian Kernel Solver for the Explicit Solution Example'''
    def __init__(self,equation):
        super(GP_Explicit_Solution_Example, self).__init__(equation)

    def y_domain(self,x_t):
        '''Compute the nonlinear term on the right at x_t'''
        return np.zeros((x_t.shape[0],1))
    
    
    def F(self,z):
        '''Compute the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        d = self.d
        sigma = self.equation.sigma()
        z_1 = z[:N_domain]
        z_2 = z[N_domain:N_domain+N_boundary]
        z_3 = z[N_domain+N_boundary:2*N_domain+N_boundary]
        z_4 = z[2*N_domain+N_boundary:3*N_domain+N_boundary]
        z_5 = z[3*N_domain+N_boundary:]
        F = np.zeros((N_domain+N_boundary,1))
        F[:N_domain] = z_4+sigma**2*z_1*z_5-(1/d+sigma**2/2)*z_5+(sigma**2/2)*z_3   
        F[N_domain:] = z_2
        return F
    
    def DF(self,z):
        '''Compute the jacobian of the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        sigma = self.equation.sigma()
        d = self.d
        z_1 = z[:N_domain]
        z_5 = z[3*N_domain+N_boundary:]
        DF = np.zeros((N_domain+N_boundary,4*N_domain+N_boundary))
        for i in range(N_domain):
            DF[i,:N_domain] = sigma**2*np.diag(z_5)
            DF[i,N_domain:N_domain+N_boundary] = 0
            DF[i,N_domain+N_boundary:2*N_domain+N_boundary] = sigma**2/2
            DF[i,2*N_domain+N_boundary:3*N_domain+N_boundary] = 1
            DF[i,3*N_domain+N_boundary:] = sigma**2*np.diag(z_1)-(1/d+sigma**2/2)

        for i in range(N_domain,N_domain+N_boundary):
            DF[i,:N_domain] = 0
            DF[i,N_domain:N_domain+N_boundary] = 1
            DF[i,N_domain+N_boundary:2*N_domain+N_boundary] = 0
            DF[i,2*N_domain+N_boundary:3*N_domain+N_boundary] = 0
            DF[i,3*N_domain+N_boundary:] = 0

        return DF
    
    def compute_PDE_loss(self, x_t_infer):
        '''Compute the PDE loss at x_t_infer'''
        sigma = self.equation.sigma()
        d = self.d
        right_vector = self.right_vector
        div_x_sol_kernel = self.div_x_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)
        laplacian_x_t_sol_kernel = self.laplacian_x_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)
        dt_x_t_sol_kernel = self.dt_x_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)
        div_x_sol = div_x_sol_kernel @ right_vector
        laplacian_x_t_sol = laplacian_x_t_sol_kernel @ right_vector
        dt_x_t_sol = dt_x_t_sol_kernel @ right_vector
        sol = self.predict(x_t_infer)
        loss = dt_x_t_sol + (sigma**2*sol-(1/d)-(sigma**2/2))*div_x_sol + (sigma**2/2)*laplacian_x_t_sol
        return loss
    

    
    
