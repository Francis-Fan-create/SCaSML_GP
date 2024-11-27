import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, hessian, jit, random, lax, value_and_grad
import jax.ops as jop
import optax

    
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
        self.nugget = 1e-2  # Regularization parameter to ensure numerical stability
    
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
    
    def dy_t_kappa(self,x_t,y_t):
        '''Compute gradient of the kernel matrix K(x_t,y_t) with respect to y_t'''
        return grad(self.kappa,argnums=1)(x_t,y_t)
    
    def dt_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the kernel matrix K(x_t,y_t) with respect to y_t'''
        dy_t_kappa = self.dy_t_kappa(x_t,y_t)
        dt_y_t_kappa = dy_t_kappa[-1]
        return dt_y_t_kappa
    
    def div_x_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to x_t'''
        dx_t_kappa = self.dx_t_kappa(x_t,y_t)
        div_x_kappa = jnp.sum(dx_t_kappa[:-1],axis=0)
        return div_x_kappa
    
    def div_y_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to y_t'''
        dy_t_kappa = self.dy_t_kappa(x_t,y_t)
        div_y_kappa = jnp.sum(dy_t_kappa[:-1],axis=0)
        return div_y_kappa
    
    def laplacian_x_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t'''
        hessian_kappa = hessian(self.kappa,argnums=0)(x_t,y_t)[:-1,:-1]
        laplacian_x_t_kappa = jnp.trace(hessian_kappa,axis1=0,axis2=1)
        return laplacian_x_t_kappa
    
    def laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the kernel matrix K(x_t,y_t) with respect to y_t'''
        hessian_kappa = hessian(self.kappa,argnums=1)(x_t,y_t)[:-1,:-1]
        laplacian_y_t_kappa = jnp.trace(hessian_kappa,axis1=0,axis2=1)
        return laplacian_y_t_kappa
    
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
        self.N_domain = N_domain
        N_boundary = x_t_boundary.shape[0]
        self.N_boundary = N_boundary
        phi_dim = 4 * N_domain + N_boundary
        self.phi_dim = phi_dim

        self.x_t_domain = x_t_domain
        self.x_t_boundary = x_t_boundary

        # Compute kernel blocks using vectorization
        # K11: Kernel between x_t_domain and x_t_domain using kappa
        K11 = jit(vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K12: Kernel between x_t_domain and x_t_boundary using kappa
        K12 = jit(vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_domain)
        # K13: Kernel between x_t_domain and x_t_domain using laplacian_y_t_kappa
        K13 = jit(vmap(lambda x_i: vmap(self.laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K14: Kernel between x_t_domain and x_t_domain using dt_y_t_kappa
        K14 = jit(vmap(lambda x_i: vmap(self.dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K15: Kernel between x_t_domain and x_t_domain using div_y_kappa
        K15 = jit(vmap(lambda x_i: vmap(self.div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)

        # K21: Kernel between x_t_boundary and x_t_domain using kappa
        K21 = jit(vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_boundary)
        # K22: Kernel between x_t_boundary and x_t_boundary using kappa
        K22 = jit(vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_boundary)
        # K23: Kernel between x_t_boundary and x_t_domain using laplacian_y_t_kappa
        K23 = jit(vmap(lambda x_i: vmap(self.laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_boundary)
        # K24: Kernel between x_t_boundary and x_t_domain using dt_y_t_kappa
        K24 = jit(vmap(lambda x_i: vmap(self.dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_boundary)
        # K25: Kernel between x_t_boundary and x_t_domain using div_y_kappa
        K25 = jit(vmap(lambda x_i: vmap(self.div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_boundary)

        # K31: Kernel between laplacian_x_t_kappa and x_t_domain
        K31 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K32: Kernel between laplacian_x_t_kappa and x_t_boundary
        K32 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_domain)
        # K33: Kernel between laplacian_x_t_kappa and laplacian_y_t_kappa
        K33 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K34: Kernel between laplacian_x_t_kappa and dt_y_t_kappa
        K34 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K35: Kernel between laplacian_x_t_kappa and div_y_kappa
        K35 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)

        # K41: Kernel between dt_x_t_kappa and x_t_domain
        K41 = jit(vmap(lambda x_i: vmap(self.dt_x_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K42: Kernel between dt_x_t_kappa and x_t_boundary
        K42 = jit(vmap(lambda x_i: vmap(self.dt_x_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_domain)
        # K43: Kernel between dt_x_t_kappa and laplacian_y_t_kappa
        K43 = jit(vmap(lambda x_i: vmap(self.dt_x_t_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K44: Kernel between dt_x_t_kappa and dt_y_t_kappa
        K44 = jit(vmap(lambda x_i: vmap(self.dt_x_t_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K45: Kernel between dt_x_t_kappa and div_y_kappa
        K45 = jit(vmap(lambda x_i: vmap(self.dt_x_t_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)

        # K51: Kernel between div_x_kappa and x_t_domain
        K51 = jit(vmap(lambda x_i: vmap(self.div_x_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K52: Kernel between div_x_kappa and x_t_boundary
        K52 = jit(vmap(lambda x_i: vmap(self.div_x_kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_domain)
        # K53: Kernel between div_x_kappa and laplacian_y_t_kappa
        K53 = jit(vmap(lambda x_i: vmap(self.div_x_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K54: Kernel between div_x_kappa and dt_y_t_kappa
        K54 = jit(vmap(lambda x_i: vmap(self.div_x_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)
        # K55: Kernel between div_x_kappa and div_y_kappa
        K55 = jit(vmap(lambda x_i: vmap(self.div_x_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_domain)

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
        # Compute the Cholesky decomposition
        cholesky_phi_phi_perturb = jnp.linalg.cholesky(kernel_phi_phi_perturb+jnp.eye(phi_dim)*self.nugget)
        if jnp.any(jnp.isnan(cholesky_phi_phi_perturb)):
            raise ValueError("Cholesky decomposition resulted in NaN values.")
        self.cholesky_phi_phi_perturb = cholesky_phi_phi_perturb    

        return kernel_phi_phi_perturb
    
    
    def kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the kernel matrix K(x_t, phi)'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        col_dim = 4 * N_domain + N_boundary  # Total columns
    
        # Compute blocks of the kernel matrix
        # K1: Kappa between x_t_infer and x_t_domain
        K1 = jit(vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain)
        # K2: Kappa between x_t_infer and x_t_boundary
        K2 = jit(vmap(lambda x_i: vmap(self.kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_infer)  # Shape: (N_infer, N_boundary)
        # K3: Laplacian_y_t_kappa between x_t_infer and x_t_domain
        K3 = jit(vmap(lambda x_i: vmap(self.laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain)
        # K4: dt_y_t_kappa between x_t_infer and x_t_domain
        K4 = jit(vmap(lambda x_i: vmap(self.dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain)
        # K5: div_y_kappa between x_t_infer and x_t_domain
        K5 = jit(vmap(lambda x_i: vmap(self.div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain)
    
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


        # Compute blocks of the gradient matrix
        # G1: dx_t_kappa between x_t_infer and x_t_domain
        G1 = jit(vmap(lambda x_i: vmap(self.dx_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain, n_input)
        # G2: dx_t_kappa between x_t_infer and x_t_boundary
        G2 = jit(vmap(lambda x_i: vmap(self.dx_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_infer)  # Shape: (N_infer, N_boundary, n_input)

        # G3: Compute gradient for laplacian_y_t_kappa
        G3 = jit(vmap(lambda x_i: vmap(grad(self.laplacian_y_t_kappa,argnums=0), in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain, n_input)

        # G4: Compute gradient for dt_y_t_kappa
        G4 = jit(vmap(lambda x_i: vmap(grad(self.dt_y_t_kappa,argnums=0), in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain, n_input)

        # G5: Compute gradient for div_y_kappa
        G5 = jit(vmap(lambda x_i: vmap(grad(self.div_y_kappa,argnums=0), in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain, n_input)

        # Concatenate blocks horizontally
        dx_t_kernel_x_phi = jnp.concatenate([G1, G2, G3, G4, G5], axis=1)  # Shape: (N_infer, col_dim, n_input)

        return dx_t_kernel_x_phi


    def laplacian_x_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the Laplacian of the kernel matrix K(x_t, phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        col_dim = 4 * N_domain + N_boundary  # Total columns


        # Compute blocks using vectorization
        # Block 1: Columns 0 to N_domain - 1
        K1 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain)

        # Block 2: Columns N_domain to N_domain + N_boundary - 1
        K2 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_infer)  # Shape: (N_infer, N_boundary)

        # Block 3: Columns N_domain + N_boundary to 2 * N_domain + N_boundary - 1
        K3 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain)

        # Block 4: Columns 2 * N_domain + N_boundary to 3 * N_domain + N_boundary - 1
        K4 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain)

        # Block 5: Columns 3 * N_domain + N_boundary to col_dim - 1
        K5 = jit(vmap(lambda x_i: vmap(self.laplacian_x_t_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)  # Shape: (N_infer, N_domain)

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
        K1 = jit(vmap(lambda x_i: vmap(self.dt_x_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)

        # Block 2: Columns N_domain to N_domain + N_boundary - 1
        K2 = jit(vmap(lambda x_i: vmap(self.dt_x_t_kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_infer)

        # Block 3: Columns N_domain + N_boundary to 2 * N_domain + N_boundary - 1
        K3 = jit(vmap(lambda x_i: vmap(self.dt_x_t_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)

        # Block 4: Columns 2 * N_domain + N_boundary to 3 * N_domain + N_boundary - 1
        K4 = jit(vmap(lambda x_i: vmap(self.dt_x_t_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)

        # Block 5: Columns 3 * N_domain + N_boundary to col_dim - 1
        K5 = jit(vmap(lambda x_i: vmap(self.dt_x_t_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)

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
        K1 = jit(vmap(lambda x_i: vmap(self.div_x_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)

        # Block 2: Columns N_domain to N_domain + N_boundary - 1
        K2 = jit(vmap(lambda x_i: vmap(self.div_x_kappa, in_axes=(None, 0))(x_i, x_t_boundary)))(x_t_infer)

        # Block 3: Columns N_domain + N_boundary to 2 * N_domain + N_boundary - 1
        K3 = jit(vmap(lambda x_i: vmap(self.div_x_laplacian_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)

        # Block 4: Columns 2 * N_domain + N_boundary to 3 * N_domain + N_boundary - 1
        K4 = jit(vmap(lambda x_i: vmap(self.div_x_dt_y_t_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)

        # Block 5: Columns 3 * N_domain + N_boundary to col_dim - 1
        K5 = jit(vmap(lambda x_i: vmap(self.div_x_div_y_kappa, in_axes=(None, 0))(x_i, x_t_domain)))(x_t_infer)

        # Concatenate blocks horizontally
        div_x_t_kernel_x_phi = jnp.hstack([K1, K2, K3, K4, K5])  # Shape: (N_infer, col_dim)

        return div_x_t_kernel_x_phi
    
    def rhs_f(self, x_t_domain):
        '''Compute the right-hand side f at x_t_domain'''
        raise NotImplementedError
    
    def bdy_g(self, x_t_boundary):
        '''Compute the boundary condition g at x_t_boundary'''
        return self.equation.g(x_t_boundary)[:,0]
    
    def time_der_rep(self, sol, rhs_f):
        '''Compute the time derivative representation on sol'''
        raise NotImplementedError
    
    def DF_domain_without_time(self, sol):
        '''Compute the derivative of sol'''
        raise NotImplementedError
    
    
    def loss_function(self, sol, rhs_f, bdy_g, L):
        '''Compute the loss function for the Gaussian process'''
        # Construct feature vector
        z_1 = sol[:self.N_domain]
        z_3 = sol[self.N_domain:2*self.N_domain]
        z_5 = sol[2*self.N_domain:]
        b = jnp.concatenate([z_1, bdy_g, z_3, self.time_der_rep(sol, rhs_f), z_5], axis=0)  # Shape: (col_dim,)
        
        # Solve the linear system
        half_vector = jnp.linalg.solve(L, b)  # Shape: (col_dim,)
        
        # Compute the loss without using .item()
        loss = jnp.dot(half_vector, half_vector)  # Scalar value
        
        return loss  # Return as a JAX scalar
    
    # def Hessian_GN(self, sol, rhs_f, bdy_g, L):
    #     '''Compute the Hessian of the loss function for the Gaussian process'''
    #     DF_domain_without_time = self.DF_domain_without_time(sol)
    #     # Compute the Hessian
    #     hess = jnp.zeros((self.phi_dim,3*self.N_domain)) # Here we compue the Hessian only w.r.t. z_1,z_3,z_5
    #     hess = hess.at[:self.N_domain, :self.N_domain].set(jnp.eye(self.N_domain))
    #     hess = hess.at[self.N_domain + self.N_boundary:2 * self.N_domain + self.N_boundary, self.N_domain:2 * self.N_domain].set(jnp.eye(self.N_domain))
    #     hess = hess.at[2 * self.N_domain + self.N_boundary:3 * self.N_domain + self.N_boundary, :].set(DF_domain_without_time)
    #     hess = hess.at[3 * self.N_domain + self.N_boundary:4 * self.N_domain + self.N_boundary, 2 * self.N_domain:].set(jnp.eye(self.N_domain))
    #     # Compute the result
    #     ss = jnp.linalg.solve(L,hess)
    #     result = 2*jnp.matmul(ss.T,ss)
    #     return result

    # def compare_hessians(self, sol, rhs_f, bdy_g, L):
    #     custom_hessian = self.Hessian_GN(sol, rhs_f, bdy_g, L)
    #     jax_hessian = hessian(self.loss_function, argnums=0)(sol, rhs_f, bdy_g, L)
    #     difference = jnp.linalg.norm(custom_hessian - jax_hessian)
    #     print(f"Hessian difference norm: {difference}")
    #     return difference
    
    # def numerical_gradient(self, sol, rhs_f, bdy_g, L, epsilon=1e-5):
    #     numerical_grad = jnp.zeros_like(sol)
    #     for i in range(sol.size):
    #         sol_plus = sol.at[i].add(epsilon)
    #         sol_minus = sol.at[i].add(-epsilon)
    #         loss_plus = self.loss_function(sol_plus, rhs_f, bdy_g, L)
    #         loss_minus = self.loss_function(sol_minus, rhs_f, bdy_g, L)
    #         numerical_grad = numerical_grad.at[i].set((loss_plus - loss_minus) / (2 * epsilon))
    #     return numerical_grad
    
    # def test_gradients(self, sol, rhs_f, bdy_g, L):
    #     analytical_grad = grad(self.loss_function)(sol, rhs_f, bdy_g, L)
    #     numerical_grad = self.numerical_gradient(sol, rhs_f, bdy_g, L)
    #     diff = jnp.linalg.norm(analytical_grad - numerical_grad)
    #     print(f"Gradient difference norm: {diff}")
    #     if diff < 1e-4:
    #         print("Gradients are correct.")
    #     else:
    #         print("Gradients mismatch. Review loss_function and Hessian_GN.")
    
    def GPsolver(self, x_t_domain, x_t_boundary, GN_steps=1000):
        '''Solve the Gaussian process using Adam optimizer from Optax with Early Stopping and Exponentially Decaying Learning Rate'''
        optimizer_steps = GN_steps
        initial_learning_rate = 1e-2
        learning_rate_decay_steps = 100  # Number of steps before each decay
        learning_rate_decay_rate = 0.96  # Decay rate
        patience = 100
        delta = 1e-5

        rhs_f = self.rhs_f(x_t_domain)
        bdy_g = self.bdy_g(x_t_boundary)

        # Define cost function
        kernel_phi_phi_perturb = self.kernel_phi_phi(x_t_domain, x_t_boundary)
        sol = random.normal(random.PRNGKey(0), (3 * self.N_domain,)) * 1e-3  # Scaled initialization
        L = self.cholesky_phi_phi_perturb

        J_hist = []
        J_now = self.loss_function(sol, rhs_f, bdy_g, L)
        J_hist.append(J_now)
        print(f"Initial loss: {J_now}")

        grad_J = grad(self.loss_function)

        # Initialize Exponentially Decaying Learning Rate Scheduler
        scheduler = optax.exponential_decay(
            init_value=initial_learning_rate,
            transition_steps=learning_rate_decay_steps,
            decay_rate=learning_rate_decay_rate,
            staircase=True
        )

        # Initialize Adam optimizer with gradient clipping and learning rate decay
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=scheduler)
        )
        opt_state = optimizer.init(sol)

        best_loss = J_now
        epochs_since_improvement = 0

        for iter_step in range(optimizer_steps):
            gradient = grad_J(sol, rhs_f, bdy_g, L)
            grad_norm = jnp.linalg.norm(gradient)
    
            # Update parameters using Adam optimizer
            updates, opt_state = optimizer.update(gradient, opt_state, sol)
            sol = optax.apply_updates(sol, updates)
    
            # Compute current loss
            J_now = self.loss_function(sol, rhs_f, bdy_g, L)
            J_hist.append(J_now)
    
            # Check for improvement
            if J_now < best_loss - delta:
                best_loss = J_now
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
    
            # Log update details
            if iter_step % 10 == 0 or iter_step == optimizer_steps - 1:
                print(f"Iteration {iter_step}: Loss = {J_now}, Gradient norm = {grad_norm}")
    
            # Early stopping based on patience
            if epochs_since_improvement >= patience:
                print(f"Early stopping at iteration {iter_step} due to no improvement in loss for {patience} steps.")
                break
    
            # Early stopping if gradient norm is small
            if grad_norm < 1e-5:
                print(f"Early stopping at iteration {iter_step} due to small gradient norm.")
                break
    

        self.loss_history = J_hist
    
        # Compute the final feature vector z
        z_1 = sol[:self.N_domain]
        z_2 = bdy_g
        z_3 = sol[self.N_domain:2 * self.N_domain]
        z_5 = sol[2 * self.N_domain:]
        z_4 = self.time_der_rep(sol, rhs_f)
        z = jnp.concatenate([z_1, z_2, z_3, z_4, z_5], axis=0)
        right_vector = jnp.linalg.solve(kernel_phi_phi_perturb, z)
        self.right_vector = right_vector[:, jnp.newaxis]
    
        sol_on_domain = self.predict(x_t_domain)
    
        return sol_on_domain
    
    # def predict(self, x_t_infer):
    #     '''Predict the solution at x_t_infer'''
    #     # Compute the kernel matrix between x_t_infer and phi
    #     kernel_x_t_phi = self.kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)  # Shape: (N_infer, col_dim)
    #     right_vector = self.right_vector  # Shape: (col_dim, 1)
    #     # Perform matrix multiplication
    #     sol_infer = kernel_x_t_phi @ right_vector  # Shape: (N_infer, 1)
    #     return sol_infer

    
    # def compute_gradient(self, x_t_infer,sol_infer):
    #     '''Compute the gradient of the solution at x_t_infer'''
    #     # Compute the gradient of the kernel matrix with respect to x_t
    #     dx_t_kernel_x_t_phi = self.dx_t_kernel_x_t_phi(x_t_infer, self.x_t_domain, self.x_t_boundary)  # Shape: (N_infer, col_dim, n_input)
    #     right_vector = self.right_vector  # Shape: (col_dim, 1)
    #     # Compute the gradient without explicit loops
    #     # Perform tensor contraction over the col_dim axis
    #     gradient = jnp.einsum('ijd,jk->ikd', dx_t_kernel_x_t_phi, right_vector)  # Shape: (N_infer, 1, n_input)
    #     # Remove the singleton dimension
    #     gradient = gradient.squeeze(axis=1)  # Shape: (N_infer, n_input)
    #     return gradient

    '''Acclerated version to compute solution and its gradient'''

    def kernel_x_t_phi_single(self, x_t):
        '''Compute the kernel vector between a single x_t and phi'''
        # Compute kernel values between x_t and x_t_domain
        kappa_values = vmap(lambda x: self.kappa(x_t, x))(self.x_t_domain)  # Shape: (N_domain,)
        # Compute kernel values between x_t and x_t_boundary
        kappa_boundary = vmap(lambda x: self.kappa(x_t, x))(self.x_t_boundary)  # Shape: (N_boundary,)
    
        # Compute other kernel terms as needed
        laplacian_kappa = vmap(lambda x: self.laplacian_y_t_kappa(x_t, x))(self.x_t_domain)  # Shape: (N_domain,)
        dt_kappa = vmap(lambda x: self.dt_y_t_kappa(x_t, x))(self.x_t_domain)  # Shape: (N_domain,)
        div_kappa = vmap(lambda x: self.div_y_kappa(x_t, x))(self.x_t_domain)  # Shape: (N_domain,)
    
        # Concatenate all kernel values to form the phi vector
        phi_x = jnp.concatenate([
            kappa_values,
            kappa_boundary,
            laplacian_kappa,
            dt_kappa,
            div_kappa
        ])  # Shape: (col_dim,)
    
        return phi_x
    
    def predict(self, x_t_infer):
        '''Predict the solution at x_t_infer using a scalar function'''
        right_vector = self.right_vector.squeeze()  # Shape: (col_dim,)
    
        # Define a function representing the solution at a single x_t point
        def solution_function(x_t):
            # Compute kernel vector between x_t and phi (training data)
            kernel_values = self.kernel_x_t_phi_single(x_t)  # Shape: (col_dim,)
            # Compute the scalar prediction
            sol = jnp.dot(kernel_values, right_vector)
            return sol
    
        # Vectorize the solution function over x_t_infer
        solution_function_vectorized = vmap(solution_function)
    
        # Compute the predictions
        sol_infer = solution_function_vectorized(x_t_infer)  # Shape: (N_infer,)
    
        return sol_infer[:,jnp.newaxis]
    
    def compute_gradient(self, x_t_infer, sol_infer):
        '''Compute the gradient of the solution at x_t_infer without large intermediate arrays'''
        right_vector = self.right_vector.squeeze()  # Shape: (col_dim,)
    
        # Define the scalar solution function as before
        def solution_function(x_t):
            kernel_values = self.kernel_x_t_phi_single(x_t)  # Shape: (col_dim,)
            sol = jnp.dot(kernel_values, right_vector)
            return sol
    
        # Compute the gradient using automatic differentiation
        gradient_function = jit(vmap(grad(solution_function)))
        gradient = gradient_function(x_t_infer)  # Shape: (N_infer, n_input)
    
        return gradient
    
    def compute_PDE_loss(self,x_t_infer):
        '''Compute the PDE loss at x_t_infer'''
        raise NotImplementedError
    
class GP_Complicated_HJB(GP):
    '''Gaussian Kernel Solver for Complicated HJB'''

    def __init__(self, equation):
        super(GP_Complicated_HJB, self).__init__(equation)

    
    def rhs_f(self, x_t):
        '''Compute the nonlinear term on the right at x_t'''
        return -2 * jnp.ones((x_t.shape[0]), dtype=x_t.dtype)

    
    def time_der_rep(self, sol, rhs_f):
        '''Compute the operator F at sol'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        d = self.d

        z_3 = sol[N_domain:2 * N_domain]
        z_5 = sol[2*N_domain:]

        rep_domain = (1 / d) * z_5 - z_3 + rhs_f
        return rep_domain

    
    def DF_domain_without_time(self, sol):
        '''Compute the Jacobian of the operator F at sol'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        d = self.d

        zeros_nd_nd = jnp.zeros((N_domain, N_domain))
        I_nd = jnp.eye(N_domain)

        DF_domain_without_time = jnp.hstack([
            zeros_nd_nd,
            -I_nd,
            (1 / d) * I_nd
        ])
 
        return DF_domain_without_time

    
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

    
    def rhs_f(self, x_t):
        '''Compute the nonlinear term on the right at x_t'''
        return jnp.zeros((x_t.shape[0]), dtype=x_t.dtype)

    
    def time_der_rep(self, sol , rhs_f):
        '''Compute the operator F at sol'''
        N_domain = self.N_domain
        d = self.d
        sigma = self.equation.sigma()

        # Extract components from z
        z_1 = sol[:N_domain]  # Shape: (N_domain, 1)
        z_3 = sol[N_domain:2 * N_domain]  # Shape: (N_domain, 1)
        z_5 = sol[2 * N_domain:]  # Shape: (N_domain, 1)

        # Compute F components
        F_domain =  -sigma**2 * z_1 * z_5 + (1 / d + sigma**2 / 2) * z_5 - (sigma**2 / 2) * z_3 + rhs_f

        return F_domain

    
    def DF_domain_without_time(self, sol):
        '''Compute the Jacobian of the operator F at sol'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        sigma = self.equation.sigma()
        d = self.d

        z_1 = sol[:N_domain]
        z_5 = sol[2 * N_domain:]

        # Compute partial derivatives
        DF_z1 = sigma**2 * jnp.diagflat(z_5)
        DF_z2 = jnp.zeros((N_domain, N_boundary))
        DF_z3 = (sigma**2 / 2) * jnp.eye(N_domain)
        DF_z4 = jnp.eye(N_domain)
        DF_z5 = sigma**2 * jnp.diagflat(z_1) - (1 / d + sigma**2 / 2) * jnp.eye(N_domain)

        # Assemble Jacobian matrix
        DF_domain_without_time = jnp.hstack([-DF_z1, -DF_z3, -DF_z5])


        return DF_domain_without_time

    
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



    
    

    

    
    
