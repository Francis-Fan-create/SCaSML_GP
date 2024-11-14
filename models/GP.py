import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, hessian, jit, random, lax
import jax.ops as jop

    
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
        self.sigma= equation.sigma()*jnp.sqrt(self.d) # Standard deviation for Gaussian kernel
        self.nugget = 1e-5  # Regularization parameter to ensure numerical stability
    
    def kappa(self,x_t,y_t):
        '''Compute the kernel entry K(x_t,y_t) for single vector x_t and y_t'''
        return jnp.exp(-jnp.sum((x_t-y_t)**2)/(2*self.sigma**2))  # Gaussian kernel
    
    def kappa_kernel(self,x_t,y_t):
        '''Compute the kernel matrix K(x_t,y_t) for batched vectors x_t and y_t'''
        N_x = x_t.shape[0]
        N_y = y_t.shape[0]
        kernel = jnp.zeros((N_x,N_y))
        for i in range(N_x):
            for j in range(N_y):
                kernel[i,j] = self.kappa(x_t[i],y_t[j])
        return kernel
    
    def dx_t_kappa(self,x_t,y_t):
        '''Compute gradient of the kernel matrix K(x_t,y_t) with respect to x_t'''
        return grad(self.kappa,argnums=0)(x_t,y_t)
    
    def dt_x_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the kernel matrix K(x_t,y_t) with respect to x_t'''
        dx_t_kappa = self.dx_t_kappa(x_t,y_t)
        dt_x_t_kappa = dx_t_kappa[-1]
        return dt_x_t_kappa
    
    def dt_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the kernel matrix K(x_t,y_t) with respect to y_t'''
        return -self.dt_x_t_kappa(x_t,y_t)
    
    def div_x_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to x_t'''
        dx_t_kappa = self.dx_t_kappa(x_t,y_t)
        div_x_kappa = jnp.sum(dx_t_kappa[:-1],axis=0)
        return div_x_kappa
    
    def div_y_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to y_t'''
        return -self.div_x_kappa(x_t,y_t)
    
    def laplacian_x_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t'''
        hessian_kappa = hessian(self.kappa,argnums=0)(x_t,y_t)[:-1,:-1]
        laplacian_x_t_kappa = jnp.trace(hessian_kappa,axis1=0,axis2=1)
        return laplacian_x_t_kappa
    
    def laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the kernel matrix K(x_t,y_t) with respect to y_t'''
        return self.laplacian_x_t_kappa(x_t,y_t)
    
    def dt_x_t_dt_y_t_kappa(self,x_t,y_t):
        '''Compute second time derivative of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        dt_x_t_dy_t_kappa = grad(self.dt_x_t_kappa,argnums=1)(x_t,y_t)
        dt_x_t_dt_y_t_kappa = dt_x_t_dy_t_kappa[-1]
        return dt_x_t_dt_y_t_kappa
    
    def dt_x_t_div_y_kappa(self,x_t,y_t):
        '''Compute time derivative of the divergence of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        dt_x_t_dy_t_kappa = grad(self.dt_x_t_kappa,argnums=1)(x_t,y_t)
        dt_x_t_div_y_kappa = jnp.sum(dt_x_t_dy_t_kappa[:-1],axis=0)
        return dt_x_t_div_y_kappa
    
    def dt_x_t_laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        hessian_dt_x_t_kappa = hessian(self.dt_x_t_kappa,argnums=1)(x_t,y_t)[:-1,:-1]
        dt_x_t_laplacian_y_t_kappa = jnp.trace(hessian_dt_x_t_kappa,axis1=0,axis2=1)
        return dt_x_t_laplacian_y_t_kappa
    
    def div_x_dt_y_t_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to x_t'''
        div_x_dy_t_kappa = grad(self.div_x_kappa,argnums=1)(x_t,y_t)
        div_x_dt_y_t_kappa = div_x_dy_t_kappa[-1]
        return div_x_dt_y_t_kappa
    
    def div_x_div_y_kappa(self,x_t,y_t):
        '''Compute divergence of the divergence of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        div_x_dy_t_kappa = grad(self.div_x_kappa,argnums=1)(x_t,y_t)
        div_x_div_y_kappa = jnp.sum(div_x_dy_t_kappa[:-1],axis=0)
        return div_x_div_y_kappa
    
    def div_x_laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute divergence of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        hessian_div_x_kappa = hessian(self.div_x_kappa,argnums=1)(x_t,y_t)[:-1,:-1]
        div_x_laplacian_y_t_kappa = jnp.trace(hessian_div_x_kappa,axis1=0,axis2=1)
        return div_x_laplacian_y_t_kappa
    
    def laplacian_x_t_dt_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        hessian_dt_y_t_kappa = hessian(self.dt_y_t_kappa,argnums=0)(x_t,y_t)[:-1,:-1]
        laplacian_x_t_dt_y_t_kappa = jnp.trace(hessian_dt_y_t_kappa,axis1=0,axis2=1)
        return laplacian_x_t_dt_y_t_kappa
    
    def laplacian_x_t_div_y_kappa(self,x_t,y_t):
        '''Compute Laplacian of the divergence of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        hessian_div_y_kappa = hessian(self.div_y_kappa,argnums=0)(x_t,y_t)[:-1,:-1]
        laplacian_x_t_div_y_kappa = jnp.trace(hessian_div_y_kappa,axis1=0,axis2=1)
        return laplacian_x_t_div_y_kappa
    
    def laplacian_x_t_laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        hessian_laplacian_y_t_kappa = hessian(self.laplacian_y_t_kappa,argnums=0)(x_t,y_t)[:-1,:-1]
        laplacian_x_t_laplacian_y_t_kappa = jnp.trace(hessian_laplacian_y_t_kappa,axis1=0,axis2=1)
        return laplacian_x_t_laplacian_y_t_kappa
    
    
    def kernel_phi_phi(self, x_t_domain, x_t_boundary):
        '''Compute the kernel matrix K(phi, phi)'''
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        phi_dim = 4 * N_domain + N_boundary

        # Prepare combined arrays for domain and boundary points
        x_t_all = jnp.concatenate([x_t_domain, x_t_boundary], axis=0)

        # Index mapping functions
        def idx_laplacian(n): return N_domain + N_boundary + n
        def idx_dt(n): return 2 * N_domain + N_boundary + n
        def idx_div(n): return 3 * N_domain + N_boundary + n

        # Compute kernel blocks using vectorization
        # K11: Kernel between x_t_domain and x_t_domain using kappa
        K11 = vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K12: Kernel between x_t_domain and x_t_boundary using kappa
        K12 = vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_domain)
        # K13: Kernel between x_t_domain and x_t_domain using laplacian_y_t_kappa
        K13 = vmap(lambda x_i: vmap(self.laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K14: Kernel between x_t_domain and x_t_domain using dt_y_t_kappa
        K14 = vmap(lambda x_i: vmap(self.dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K15: Kernel between x_t_domain and x_t_domain using div_y_kappa
        K15 = vmap(lambda x_i: vmap(self.div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)

        # K21: Kernel between x_t_boundary and x_t_domain using kappa
        K21 = vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_boundary)
        # K22: Kernel between x_t_boundary and x_t_boundary using kappa
        K22 = vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_boundary)
        # K23: Kernel between x_t_boundary and x_t_domain using laplacian_y_t_kappa
        K23 = vmap(lambda x_i: vmap(self.laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_boundary)
        # K24: Kernel between x_t_boundary and x_t_domain using dt_y_t_kappa
        K24 = vmap(lambda x_i: vmap(self.dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_boundary)
        # K25: Kernel between x_t_boundary and x_t_domain using div_y_kappa
        K25 = vmap(lambda x_i: vmap(self.div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_boundary)

        # K31: Kernel between laplacian_x_t_kappa and x_t_domain
        K31 = vmap(lambda x_i: vmap(self.laplacian_x_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K32: Kernel between laplacian_x_t_kappa and x_t_boundary
        K32 = vmap(lambda x_i: vmap(self.laplacian_x_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_domain)
        # K33: Kernel between laplacian_x_t_kappa and laplacian_y_t_kappa
        K33 = vmap(lambda x_i: vmap(self.laplacian_x_t_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K34: Kernel between laplacian_x_t_kappa and dt_y_t_kappa
        K34 = vmap(lambda x_i: vmap(self.laplacian_x_t_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K35: Kernel between laplacian_x_t_kappa and div_y_kappa
        K35 = vmap(lambda x_i: vmap(self.laplacian_x_t_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)

        # K41: Kernel between dt_x_t_kappa and x_t_domain
        K41 = vmap(lambda x_i: vmap(self.dt_x_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K42: Kernel between dt_x_t_kappa and x_t_boundary
        K42 = vmap(lambda x_i: vmap(self.dt_x_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_domain)
        # K43: Kernel between dt_x_t_kappa and laplacian_y_t_kappa
        K43 = vmap(lambda x_i: vmap(self.dt_x_t_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K44: Kernel between dt_x_t_kappa and dt_y_t_kappa
        K44 = vmap(lambda x_i: vmap(self.dt_x_t_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K45: Kernel between dt_x_t_kappa and div_y_kappa
        K45 = vmap(lambda x_i: vmap(self.dt_x_t_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)

        # K51: Kernel between div_x_kappa and x_t_domain
        K51 = vmap(lambda x_i: vmap(self.div_x_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K52: Kernel between div_x_kappa and x_t_boundary
        K52 = vmap(lambda x_i: vmap(self.div_x_kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_domain)
        # K53: Kernel between div_x_kappa and laplacian_y_t_kappa
        K53 = vmap(lambda x_i: vmap(self.div_x_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K54: Kernel between div_x_kappa and dt_y_t_kappa
        K54 = vmap(lambda x_i: vmap(self.div_x_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)
        # K55: Kernel between div_x_kappa and div_y_kappa
        K55 = vmap(lambda x_i: vmap(self.div_x_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_domain)

        # Assemble the full kernel matrix
        row1 = jnp.hstack([K11, K12, K13, K14, K15])  # Shape: (N_domain, phi_dim)
        row2 = jnp.hstack([K21, K22, K23, K24, K25])  # Shape: (N_boundary, phi_dim)
        row3 = jnp.hstack([K31, K32, K33, K34, K35])  # Shape: (N_domain, phi_dim)
        row4 = jnp.hstack([K41, K42, K43, K44, K45])  # Shape: (N_domain, phi_dim)
        row5 = jnp.hstack([K51, K52, K53, K54, K55])  # Shape: (N_domain, phi_dim)

        # Vertically stack the rows to form the full matrix
        kernel_phi_phi = jnp.vstack([row1, row2, row3, row4, row5])

        # Compute traces except the second feature in each cycle
        trace_domain = jnp.trace(K11)
        trace_laplacian = jnp.trace(K33)
        trace_dt = jnp.trace(K44)
        trace_div = jnp.trace(K55)

        # Scale them with the laplacian trace
        eigen_nugget = self.nugget * jnp.concatenate([
            trace_domain * jnp.ones(N_domain),
            jnp.ones(N_boundary),
            jnp.ones(N_domain),
            trace_dt * jnp.ones(N_domain),
            trace_div * jnp.ones(N_domain)
        ])
        eigen_nugget = eigen_nugget / trace_laplacian
        kernel_phi_phi_perturb = kernel_phi_phi + jnp.diag(eigen_nugget)

        return kernel_phi_phi_perturb
    
    
    def kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the kernel matrix K(x_t, phi)'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        col_dim = 4 * N_domain + N_boundary  # Total columns
    
        # Compute blocks of the kernel matrix
        # K1: Kappa between x_t_infer and x_t_domain
        K1 = vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
        # K2: Kappa between x_t_infer and x_t_boundary
        K2 = vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_infer)  # Shape: (N_infer, N_boundary)
        # K3: Laplacian_y_t_kappa between x_t_infer and x_t_domain
        K3 = vmap(lambda x_i: vmap(self.laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
        # K4: dt_y_t_kappa between x_t_infer and x_t_domain
        K4 = vmap(lambda x_i: vmap(self.dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
        # K5: div_y_kappa between x_t_infer and x_t_domain
        K5 = vmap(lambda x_i: vmap(self.div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
    
        # Concatenate blocks horizontally to form the full matrix
        kernel_x_phi = jnp.hstack([K1, K2, K3, K4, K5])  # Shape: (N_infer, col_dim)
    
        return kernel_x_phi
    
    
    def dx_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the gradient of the kernel matrix K(x_t, phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        n_input = self.n_input
        col_dim = 4 * N_domain + N_boundary  # Total columns
    
        sigma_squared_inv = -1 / (self.sigma ** 2)
    
        # Compute differences between x_t_infer and x_t_domain/boundary
        delta_x_domain = x_t_infer[:, None, :] - x_t_domain[None, :, :]  # Shape: (N_infer, N_domain, n_input)
        delta_x_boundary = x_t_infer[:, None, :] - x_t_boundary[None, :, :]  # Shape: (N_infer, N_boundary, n_input)
    
        # Compute blocks of the gradient matrix
        # G1: dx_t_kappa between x_t_infer and x_t_domain
        G1 = vmap(lambda x_i: vmap(self.dx_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain, n_input)
        # G2: dx_t_kappa between x_t_infer and x_t_boundary
        G2 = vmap(lambda x_i: vmap(self.dx_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_infer)  # Shape: (N_infer, N_boundary, n_input)
    
        # G3: Compute gradient for laplacian_y_t_kappa
        laplacian_y = vmap(lambda x_i: vmap(self.laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
        G3 = sigma_squared_inv * delta_x_domain * laplacian_y[:, :, None]  # Shape: (N_infer, N_domain, n_input)
    
        # G4: Compute gradient for dt_y_t_kappa
        dt_y = vmap(lambda x_i: vmap(self.dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
        G4 = sigma_squared_inv * delta_x_domain * dt_y[:, :, None]  # Shape: (N_infer, N_domain, n_input)
    
        # G5: Compute gradient for div_y_kappa
        div_y = vmap(lambda x_i: vmap(self.div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
        G5 = sigma_squared_inv * delta_x_domain * div_y[:, :, None]  # Shape: (N_infer, N_domain, n_input)
    
        # Concatenate blocks horizontally
        dx_t_kernel_x_phi = jnp.concatenate([G1, G2, G3, G4, G5], axis=1)  # Shape: (N_infer, col_dim, n_input)
    
        return dx_t_kernel_x_phi
    
    
    def laplacian_x_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the Laplacian of the kernel matrix K(x_t, phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        col_dim = 4 * N_domain + N_boundary  # Total columns
    
        # Precompute indices for convenience
        idx1 = N_domain
        idx2 = N_domain + N_boundary
        idx3 = 2 * N_domain + N_boundary
        idx4 = 3 * N_domain + N_boundary
    
        # Compute blocks using vectorization
        # Block 1: Columns 0 to N_domain - 1
        K1 = vmap(lambda x_i: vmap(self.laplacian_x_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
    
        # Block 2: Columns N_domain to N_domain + N_boundary - 1
        K2 = vmap(lambda x_i: vmap(self.laplacian_x_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_infer)  # Shape: (N_infer, N_boundary)
    
        # Block 3: Columns N_domain + N_boundary to 2 * N_domain + N_boundary - 1
        K3 = vmap(lambda x_i: vmap(self.laplacian_x_t_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
    
        # Block 4: Columns 2 * N_domain + N_boundary to 3 * N_domain + N_boundary - 1
        K4 = vmap(lambda x_i: vmap(self.laplacian_x_t_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
    
        # Block 5: Columns 3 * N_domain + N_boundary to col_dim - 1
        K5 = vmap(lambda x_i: vmap(self.laplacian_x_t_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)  # Shape: (N_infer, N_domain)
    
        # Concatenate blocks horizontally to form the full matrix
        laplacian_x_t_kernel_x_phi = jnp.hstack([K1, K2, K3, K4, K5])  # Shape: (N_infer, col_dim)
    
        return laplacian_x_t_kernel_x_phi
    
    
    def dt_x_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the time derivative of the kernel matrix K(x_t, phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        col_dim = 4 * N_domain + N_boundary  # Total columns
    
        # Compute blocks using vectorization
        # Block 1: Columns 0 to N_domain - 1
        K1 = vmap(lambda x_i: vmap(self.dt_x_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)
    
        # Block 2: Columns N_domain to N_domain + N_boundary - 1
        K2 = vmap(lambda x_i: vmap(self.dt_x_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_infer)
    
        # Block 3: Columns N_domain + N_boundary to 2 * N_domain + N_boundary - 1
        K3 = vmap(lambda x_i: vmap(self.dt_x_t_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)
    
        # Block 4: Columns 2 * N_domain + N_boundary to 3 * N_domain + N_boundary - 1
        K4 = vmap(lambda x_i: vmap(self.dt_x_t_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)
    
        # Block 5: Columns 3 * N_domain + N_boundary to col_dim - 1
        K5 = vmap(lambda x_i: vmap(self.dt_x_t_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)
    
        # Concatenate blocks horizontally
        dt_x_t_kernel_x_phi = jnp.hstack([K1, K2, K3, K4, K5])  # Shape: (N_infer, col_dim)
    
        return dt_x_t_kernel_x_phi
    
    
    def div_x_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the divergence of the kernel matrix K(x_t, phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        col_dim = 4 * N_domain + N_boundary  # Total columns
    
        # Compute blocks using vectorization
        # Block 1: Columns 0 to N_domain - 1
        K1 = vmap(lambda x_i: vmap(self.div_x_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)
    
        # Block 2: Columns N_domain to N_domain + N_boundary - 1
        K2 = vmap(lambda x_i: vmap(self.div_x_kappa, in_axes=(None, 0))(x_i, x_t_boundary))(x_t_infer)
    
        # Block 3: Columns N_domain + N_boundary to 2 * N_domain + N_boundary - 1
        K3 = vmap(lambda x_i: vmap(self.div_x_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)
    
        # Block 4: Columns 2 * N_domain + N_boundary to 3 * N_domain + N_boundary - 1
        K4 = vmap(lambda x_i: vmap(self.div_x_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)
    
        # Block 5: Columns 3 * N_domain + N_boundary to col_dim - 1
        K5 = vmap(lambda x_i: vmap(self.div_x_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain))(x_t_infer)
    
        # Concatenate blocks horizontally
        div_x_t_kernel_x_phi = jnp.hstack([K1, K2, K3, K4, K5])  # Shape: (N_infer, col_dim)
    
        return div_x_t_kernel_x_phi
    
    def y_domain(self,x_t):
        '''Compute the nonlinear term on the right at x_t'''
        raise NotImplementedError
    
    def y_boundary(self,x_t):
        '''Compute the terminal condition at x_t'''
        return self.equation.g(x_t)[:,jnp.newaxis]
    
    
    def GPsolver(self, x_t_domain, x_t_boundary, GN_step=13, key=0):
        '''Solve the Gaussian Process for the given domain and boundary points for linear PDE'''
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        self.x_t_domain = x_t_domain
        self.x_t_boundary = x_t_boundary
        self.N_domain = N_domain
        self.N_boundary = N_boundary
    
        # Initialize random vector z_k using JAX's random number generator with PRNGKey
        key = random.PRNGKey(key)
        z_k = random.uniform(key, shape=(4 * N_domain + N_boundary, 1), minval=-1.0, maxval=1.0)
    
        # Concatenate y_domain and y_boundary to form the target vector y
        y = jnp.concatenate((self.y_domain(x_t_domain), self.y_boundary(x_t_boundary)), axis=0)
    
        # Compute the perturbed kernel matrix K_phi_phi
        kernel_phi_phi_perturb = self.kernel_phi_phi(x_t_domain, x_t_boundary)
    
        # Define the body function for the Gauss-Newton iteration
        def body_fun(i, z_k):
            '''
            Gauss-Newton iteration body function.
    
            Args:
                i: Iteration index (unused).
                z_k: Current estimate of z.
    
            Returns:
                Updated estimate of z.
            '''
            DF_k = self.DF(z_k)
            # Compute the projected kernel matrix
            proj_kernel_phi_phi = DF_k @ kernel_phi_phi_perturb @ DF_k.T
            # Compute the residual
            residual = y - self.F(z_k) + DF_k @ z_k
            # Solve for gamma using linear solver
            gamma = jnp.linalg.solve(proj_kernel_phi_phi, residual)
            # Update z_k
            z_k_new = kernel_phi_phi_perturb @ DF_k.T @ gamma
            return z_k_new
    
        # Perform Gauss-Newton iterations using JAX's lax.fori_loop for JIT compatibility
        z_k = lax.fori_loop(0, GN_step, body_fun, z_k)
    
        # Solve for the right vector using the perturbed kernel matrix
        right_vector = jnp.linalg.solve(kernel_phi_phi_perturb, z_k)
        self.right_vector = right_vector
    
        # Compute the solution on the domain
        sol_on_domain = self.kernel_x_t_phi(x_t_domain, x_t_domain, x_t_boundary) @ right_vector
        return sol_on_domain
    
    
    def predict(self, x_t_infer):
        '''Predict the solution at x_t_infer'''
        # Compute the kernel matrix between x_t_infer and phi
        kernel_x_t_phi = self.kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)  # Shape: (N_infer, col_dim)
        right_vector = self.right_vector  # Shape: (col_dim, 1)
        # Perform matrix multiplication
        sol_infer = kernel_x_t_phi @ right_vector  # Shape: (N_infer, 1)
        return sol_infer

    
    def compute_gradient(self, x_t_infer,sol_infer):
        '''Compute the gradient of the solution at x_t_infer'''
        # Compute the gradient of the kernel matrix with respect to x_t
        dx_t_kernel_x_t_phi = self.dx_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)  # Shape: (N_infer, col_dim, n_input)
        right_vector = self.right_vector  # Shape: (col_dim, 1)
        # Compute the gradient without explicit loops
        # Perform tensor contraction over the col_dim axis
        gradient = jnp.einsum('ijd,jk->ikd', dx_t_kernel_x_t_phi, right_vector)  # Shape: (N_infer, 1, n_input)
        # Remove the singleton dimension
        gradient = gradient.squeeze(axis=1)  # Shape: (N_infer, n_input)
        return gradient
    
    def compute_PDE_loss(self,x_t_infer):
        '''Compute the PDE loss at x_t_infer'''
        raise NotImplementedError
    
class GP_Complicated_HJB(GP):
    '''Gaussian Kernel Solver for Complicated HJB'''

    def __init__(self, equation):
        super(GP_Complicated_HJB, self).__init__(equation)

    
    def y_domain(self, x_t):
        '''Compute the nonlinear term on the right at x_t'''
        return -2 * jnp.ones((x_t.shape[0], 1), dtype=x_t.dtype)

    
    def F(self, z):
        '''Compute the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        d = self.d

        z_2 = z[N_domain:N_domain + N_boundary]
        z_3 = z[N_domain + N_boundary:2 * N_domain + N_boundary]
        z_4 = z[2 * N_domain + N_boundary:3 * N_domain + N_boundary]
        z_5 = z[3 * N_domain + N_boundary:]

        F_domain = z_4 - (1 / d) * z_5 + z_3
        F_boundary = z_2
        F = jnp.concatenate([F_domain, F_boundary], axis=0)
        return F

    
    def DF(self, z):
        '''Compute the Jacobian of the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        d = self.d

        zeros_nd_nd = jnp.zeros((N_domain, N_domain))
        zeros_nd_nb = jnp.zeros((N_domain, N_boundary))
        I_nd = jnp.eye(N_domain)
        I_nb = jnp.eye(N_boundary)

        DF_upper = jnp.hstack([
            zeros_nd_nd,
            zeros_nd_nb,
            I_nd,
            I_nd,
            (-1 / d) * I_nd
        ])

        zeros_nb_nd = jnp.zeros((N_boundary, N_domain))
        zeros_nb_rest = jnp.zeros((N_boundary, 3 * N_domain))

        DF_lower = jnp.hstack([
            zeros_nb_nd,
            I_nb,
            zeros_nb_rest
        ])

        DF = jnp.vstack([DF_upper, DF_lower])
        return DF

    
    def compute_PDE_loss(self, x_t_infer):
        '''Compute the PDE loss at x_t_infer'''
        d = self.d
        right_vector = self.right_vector

        dt_x_t_sol_kernel = self.dt_x_t_kernel_x_t_phi(
            x_t_infer, self.x_t_domain, self.x_t_boundary)
        div_x_sol_kernel = self.div_x_t_kernel_x_t_phi(
            x_t_infer, self.x_t_domain, self.x_t_boundary)
        laplacian_x_t_sol_kernel = self.laplacian_x_t_kernel_x_t_phi(
            x_t_infer, self.x_t_domain, self.x_t_boundary)

        dt_x_t_sol = dt_x_t_sol_kernel @ right_vector
        div_x_sol = div_x_sol_kernel @ right_vector
        laplacian_x_t_sol = laplacian_x_t_sol_kernel @ right_vector

        loss = dt_x_t_sol - (1 / d) * div_x_sol + laplacian_x_t_sol + 2
        return loss

class GP_Explicit_Solution_Example(GP):
    '''Gaussian Kernel Solver for the Explicit Solution Example'''

    def __init__(self, equation):
        super(GP_Explicit_Solution_Example, self).__init__(equation)

    
    def y_domain(self, x_t):
        '''Compute the nonlinear term on the right at x_t'''
        return jnp.zeros((x_t.shape[0], 1), dtype=x_t.dtype)

    
    def F(self, z):
        '''Compute the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        d = self.d
        sigma = self.equation.sigma()

        # Extract components from z
        z_1 = z[:N_domain]  # Shape: (N_domain, 1)
        z_2 = z[N_domain:N_domain + N_boundary]  # Shape: (N_boundary, 1)
        z_3 = z[N_domain + N_boundary:2 * N_domain + N_boundary]  # Shape: (N_domain, 1)
        z_4 = z[2 * N_domain + N_boundary:3 * N_domain + N_boundary]  # Shape: (N_domain, 1)
        z_5 = z[3 * N_domain + N_boundary:]  # Shape: (N_domain, 1)

        # Compute F components
        F_domain = z_4 + sigma**2 * z_1 * z_5 - (1 / d + sigma**2 / 2) * z_5 + (sigma**2 / 2) * z_3
        F_boundary = z_2

        # Combine F components
        F = jnp.concatenate([F_domain, F_boundary], axis=0)
        return F

    
    def DF(self, z):
        '''Compute the Jacobian of the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        sigma = self.equation.sigma()
        d = self.d

        z_1 = z[:N_domain]
        z_5 = z[3 * N_domain + N_boundary:]

        # Compute partial derivatives
        DF_z1 = sigma**2 * jnp.diagflat(z_5)
        DF_z2 = jnp.zeros((N_domain, N_boundary))
        DF_z3 = (sigma**2 / 2) * jnp.eye(N_domain)
        DF_z4 = jnp.eye(N_domain)
        DF_z5 = sigma**2 * jnp.diagflat(z_1) - (1 / d + sigma**2 / 2) * jnp.eye(N_domain)

        # Assemble Jacobian matrix
        DF_upper = jnp.hstack([DF_z1, DF_z2, DF_z3, DF_z4, DF_z5])
        DF_lower = jnp.hstack([
            jnp.zeros((N_boundary, N_domain)),        # ∂F_boundary/∂z_1
            jnp.eye(N_boundary),                      # ∂F_boundary/∂z_2
            jnp.zeros((N_boundary, 3 * N_domain))     # ∂F_boundary/∂z_3, ∂F_boundary/∂z_4, ∂F_boundary/∂z_5
        ])

        DF = jnp.vstack([DF_upper, DF_lower])
        return DF

    
    def compute_PDE_loss(self, x_t_infer):
        '''Compute the PDE loss at x_t_infer'''
        sigma = self.equation.sigma()
        d = self.d
        right_vector = self.right_vector

        # Compute kernel matrices
        div_x_sol_kernel = self.div_x_t_kernel_x_t_phi(
            x_t_infer, self.x_t_domain, self.x_t_boundary)
        laplacian_x_sol_kernel = self.laplacian_x_t_kernel_x_t_phi(
            x_t_infer, self.x_t_domain, self.x_t_boundary)
        dt_x_sol_kernel = self.dt_x_t_kernel_x_t_phi(
            x_t_infer, self.x_t_domain, self.x_t_boundary)

        # Compute solutions
        div_x_sol = div_x_sol_kernel @ right_vector
        laplacian_x_sol = laplacian_x_sol_kernel @ right_vector
        dt_x_sol = dt_x_sol_kernel @ right_vector
        sol = self.predict(x_t_infer)

        # Compute PDE loss
        loss = dt_x_sol + (sigma**2 * sol - (1 / d) - (sigma**2 / 2)) * div_x_sol \
            + (sigma**2 / 2) * laplacian_x_sol
        return loss



    
    

    

    
    
