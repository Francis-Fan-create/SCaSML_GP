import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, hessian, jit, random, lax, value_and_grad, jvp
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

    def laplacian_op(self, f):
        '''Compute the Laplacian of a function f using Hutchinson's method'''
        MC = 5
        def hvp(f, x, i):
            call_jei = lambda x:jit(grad(f))(x)[i]
            return jit(grad(call_jei))(x)
        def laplacian(x):
            idx_set = random.choice(random.PRNGKey(0), self.d, shape=(MC,), replace=False)
            f_hess_diag_fn = lambda i: hvp(f, x, i)[i]
            hess_diag_val = jit(vmap(f_hess_diag_fn))(idx_set)
            return jnp.mean(hess_diag_val)*self.d
        return laplacian
    
    def kappa(self,x_t,y_t):
        '''Compute the kernel entry K(x_t,y_t) for single vector x_t and y_t'''
        return jnp.exp(-jnp.sum((x_t-y_t)**2)/(2*self.sigma**2)).astype(jnp.float16)  # Gaussian kernel
    
    def kappa_kernel(self,x_t,y_t):
        '''Compute the kernel matrix K(x_t,y_t) for batched vectors x_t and y_t'''
        N_x = x_t.shape[0]
        N_y = y_t.shape[0]
        kernel = jnp.zeros((N_x,N_y),dtype=jnp.float16)
        for i in range(N_x):
            for j in range(N_y):
                kernel = kernel.at[i,j].set(self.kappa(x_t[i], y_t[j]))
        return kernel.astype(jnp.float16)
    
    def dx_t_kappa(self,x_t,y_t):
        '''Compute gradient of the kernel matrix K(x_t,y_t) with respect to x_t'''
        return grad(self.kappa,argnums=0)(x_t,y_t).astype(jnp.float16) 
    
    def dt_x_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the kernel matrix K(x_t,y_t) with respect to x_t'''
        dx_t_kappa = self.dx_t_kappa(x_t,y_t)
        dt_x_t_kappa = dx_t_kappa[-1]
        return dt_x_t_kappa.astype(jnp.float16)
    
    def dy_t_kappa(self,x_t,y_t):
        '''Compute gradient of the kernel matrix K(x_t,y_t) with respect to y_t'''
        return grad(self.kappa,argnums=1)(x_t,y_t).astype(jnp.float16) 
    
    def dt_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the kernel matrix K(x_t,y_t) with respect to y_t'''
        dy_t_kappa = self.dy_t_kappa(x_t,y_t)
        dt_y_t_kappa = dy_t_kappa[-1]
        return dt_y_t_kappa.astype(jnp.float16)
    
    def div_x_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to x_t'''
        dx_t_kappa = self.dx_t_kappa(x_t,y_t)
        div_x_kappa = jnp.sum(dx_t_kappa[:-1],axis=0)
        return div_x_kappa.astype(jnp.float16)
    
    def div_y_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to y_t'''
        dy_t_kappa = self.dy_t_kappa(x_t,y_t)
        div_y_kappa = jnp.sum(dy_t_kappa[:-1],axis=0)
        return div_y_kappa.astype(jnp.float16)
    
    def laplacian_x_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t'''
        # hessian_kappa = hessian(self.kappa,argnums=0)(x_t,y_t)[:-1,:-1]
        # laplacian_x_t_kappa = jnp.trace(hessian_kappa,axis1=0,axis2=1)
        t_x = x_t[0,jnp.newaxis]
        x = x_t[1:]
        kappa_x_t = lambda x: self.kappa(jnp.concatenate((x,t_x)),y_t) # Compute kernel with x only
        laplacian_x_t_kappa = self.laplacian_op(kappa_x_t)
        return laplacian_x_t_kappa(x).astype(jnp.float16) 
    
    def laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the kernel matrix K(x_t,y_t) with respect to y_t'''
        # hessian_kappa = hessian(self.kappa,argnums=1)(x_t,y_t)[:-1,:-1]
        # laplacian_y_t_kappa = jnp.trace(hessian_kappa,axis1=0,axis2=1)
        t_y = y_t[0,jnp.newaxis]
        y = y_t[1:]
        kappa_y_t = lambda y: self.kappa(x_t,jnp.concatenate((y,t_y))) # Compute kernel with y only
        laplacian_y_t_kappa = self.laplacian_op(kappa_y_t)
        return laplacian_y_t_kappa(y).astype(jnp.float16) 
    
    def dt_x_t_dt_y_t_kappa(self,x_t,y_t):
        '''Compute second time derivative of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        dt_x_t_dy_t_kappa = grad(self.dt_x_t_kappa,argnums=1)(x_t,y_t).astype(jnp.float16) 
        dt_x_t_dt_y_t_kappa = dt_x_t_dy_t_kappa[-1]
        return dt_x_t_dt_y_t_kappa.astype(jnp.float16)
    
    def dt_x_t_div_y_kappa(self,x_t,y_t):
        '''Compute time derivative of the divergence of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        dt_x_t_dy_t_kappa = grad(self.dt_x_t_kappa,argnums=1)(x_t,y_t).astype(jnp.float16) 
        dt_x_t_div_y_kappa = jnp.sum(dt_x_t_dy_t_kappa[:-1],axis=0)
        return dt_x_t_div_y_kappa.astype(jnp.float16)
    
    def dt_x_t_laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        # hessian_dt_x_t_kappa = hessian(self.dt_x_t_kappa,argnums=1)(x_t,y_t)[:-1,:-1]
        # dt_x_t_laplacian_y_t_kappa = jnp.trace(hessian_dt_x_t_kappa,axis1=0,axis2=1)
        t_y = y_t[0,jnp.newaxis]
        y = y_t[1:]
        dt_x_t_kappa_y_t = lambda y: self.dt_x_t_kappa(x_t, jnp.concatenate((y,t_y))) # Compute kernel with x only
        dt_x_t_laplacian_y_t_kappa = self.laplacian_op(dt_x_t_kappa_y_t)
        return dt_x_t_laplacian_y_t_kappa(y).astype(jnp.float16) 
    
    def div_x_dt_y_t_kappa(self,x_t,y_t):
        '''Compute divergence of the kernel matrix K(x_t,y_t) with respect to x_t'''
        div_x_dy_t_kappa = grad(self.div_x_kappa,argnums=1)(x_t,y_t).astype(jnp.float16) 
        div_x_dt_y_t_kappa = div_x_dy_t_kappa[-1]
        return div_x_dt_y_t_kappa.astype(jnp.float16)
    
    def div_x_div_y_kappa(self,x_t,y_t):
        '''Compute divergence of the divergence of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        div_x_dy_t_kappa = grad(self.div_x_kappa,argnums=1)(x_t,y_t).astype(jnp.float16) 
        div_x_div_y_kappa = jnp.sum(div_x_dy_t_kappa[:-1],axis=0)
        return div_x_div_y_kappa.astype(jnp.float16)
    
    def div_x_laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute divergence of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        # hessian_div_x_kappa = hessian(self.div_x_kappa,argnums=1)(x_t,y_t)[:-1,:-1]
        # div_x_laplacian_y_t_kappa = jnp.trace(hessian_div_x_kappa,axis1=0,axis2=1)
        t_y = y_t[0,jnp.newaxis]
        y = y_t[1:]
        div_x_kappa_y_t = lambda y: self.div_x_kappa(x_t, jnp.concatenate((y,t_y))) # Compute kernel with x only
        div_x_laplacian_y_t_kappa = self.laplacian_op(div_x_kappa_y_t)
        return div_x_laplacian_y_t_kappa(y).astype(jnp.float16) 
    
    def laplacian_x_t_dt_y_t_kappa(self,x_t,y_t):
        '''Compute time derivative of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        # hessian_dt_y_t_kappa = hessian(self.dt_y_t_kappa,argnums=0)(x_t,y_t)[:-1,:-1]
        # laplacian_x_t_dt_y_t_kappa = jnp.trace(hessian_dt_y_t_kappa,axis1=0,axis2=1)
        t_x = x_t[0,jnp.newaxis]
        x = x_t[1:]
        dt_y_t_kappa = lambda x: self.dt_y_t_kappa(jnp.concatenate((x,t_x)),y_t) # Compute kernel with x only
        laplacian_x_t_dt_y_t_kappa = self.laplacian_op(dt_y_t_kappa)
        return laplacian_x_t_dt_y_t_kappa(x).astype(jnp.float16) 
    
    def laplacian_x_t_div_y_kappa(self,x_t,y_t):
        '''Compute Laplacian of the divergence of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        # hessian_div_y_kappa = hessian(self.div_y_kappa,argnums=0)(x_t,y_t)[:-1,:-1]
        # laplacian_x_t_div_y_kappa = jnp.trace(hessian_div_y_kappa,axis1=0,axis2=1)
        t_x = x_t[0,jnp.newaxis]
        x = x_t[1:]
        div_y_kappa = lambda x: self.div_y_kappa(jnp.concatenate((x,t_x)),y_t) # Compute kernel with x only
        laplacian_x_t_div_y_kappa = self.laplacian_op(div_y_kappa)
        return laplacian_x_t_div_y_kappa(x).astype(jnp.float16) 
    
    def laplacian_x_t_laplacian_y_t_kappa(self,x_t,y_t):
        '''Compute Laplacian of the Laplacian of the kernel matrix K(x_t,y_t) with respect to x_t then y_t'''
        # hessian_laplacian_y_t_kappa = hessian(self.laplacian_y_t_kappa,argnums=0)(x_t,y_t)[:-1,:-1]
        # laplacian_x_t_laplacian_y_t_kappa = jnp.trace(hessian_laplacian_y_t_kappa,axis1=0,axis2=1)
        t_x = x_t[0,jnp.newaxis]
        x = x_t[1:]
        laplacian_y_t_kappa = lambda x: self.laplacian_y_t_kappa(jnp.concatenate((x,t_x)),y_t) # Compute kernel with x only
        laplacian_x_t_laplacian_y_t_kappa = self.laplacian_op(laplacian_y_t_kappa)
        return laplacian_x_t_laplacian_y_t_kappa(x).astype(jnp.float16) 
    
    
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
        kernel_phi_phi = jnp.vstack([row1, row2, row3, row4, row5]).astype(jnp.float64)  # Shape: (phi_dim, phi_dim)
        # SVD decomposition, acclearated by jit
        U, S, Vt = jit(jnp.linalg.svd)(kernel_phi_phi)
        S_perturb = S + self.nugget
        # Compute the Cholesky decomposition
        cholesky_phi_phi_perturb = U @ jnp.diag(jnp.sqrt(S_perturb))
        if jnp.any(jnp.isnan(cholesky_phi_phi_perturb)):
            raise ValueError("Cholesky decomposition resulted in NaN values.")
        self.cholesky_phi_phi_perturb = cholesky_phi_phi_perturb.astype(jnp.float16)    
        kernel_phi_phi_perturb = cholesky_phi_phi_perturb @ cholesky_phi_phi_perturb.T
        return kernel_phi_phi_perturb.astype(jnp.float16)
    
    
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
    
        return kernel_x_phi.astype(jnp.float16)
    
    
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

        return dx_t_kernel_x_phi.astype(jnp.float16)


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

        return laplacian_x_t_kernel_x_phi.astype(jnp.float16)


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

        return dt_x_t_kernel_x_phi.astype(jnp.float16)


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

        return div_x_t_kernel_x_phi.astype(jnp.float16)
    
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
        
        return loss.astype(jnp.float16)  # Return as a JAX scalar
    
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
    
    def GPsolver(self, x_t_domain, x_t_boundary, GN_steps=20):
        '''Solve the Gaussian process using Newton's method with line search and regularization'''
        optimizer_steps = GN_steps
        initial_damping = 1e-4  # Initial regularization parameter
        max_damping = 1.0  # Maximum regularization parameter
        damping_factor = 10.0  # Regularization growth factor
        damping_decrease = 0.1  # Regularization reduction factor
        
        # Calculate right-hand side and boundary conditions
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
        
        # Define gradient and Hessian functions
        grad_J = grad(self.loss_function)
        hess_J = hessian(self.loss_function)
        
        damping = initial_damping
        
        for iter_step in range(optimizer_steps):
            # Calculate current gradient
            gradient = grad_J(sol, rhs_f, bdy_g, L)
            grad_norm = jnp.linalg.norm(gradient)
            
            # Early stopping if gradient norm is small
            if grad_norm < 1e-5:
                print(f"Early stopping at iteration {iter_step} due to small gradient norm.")
                break
            
            # Calculate Hessian matrix
            hessian_matrix = hess_J(sol, rhs_f, bdy_g, L)
            
            # Add regularization to ensure positive definiteness of Hessian
            regularized_hessian = hessian_matrix + damping * jnp.eye(hessian_matrix.shape[0])
            
            # Solve for Newton direction: H * Δx = -g
            try:
                newton_direction = jnp.linalg.solve(regularized_hessian, -gradient)
            except:
                # If matrix solve fails, increase regularization and try again
                damping = min(damping * damping_factor, max_damping)
                regularized_hessian = hessian_matrix + damping * jnp.eye(hessian_matrix.shape[0])
                newton_direction = jnp.linalg.solve(regularized_hessian, -gradient)
            
            # Line search to determine step size
            alpha = 1.0
            # beta = 0.5  # Step size decay factor
            # c = 1e-4   # Armijo condition constant
            # max_line_search_iters = 10
            
            # # Calculate directional derivative for current direction
            # directional_derivative = jnp.dot(gradient, newton_direction)
            
            # # Line search: Armijo condition
            # current_loss = J_now
            
            # # Standard Python loop for line search
            # found_step = False
            # line_search_iters = 0
            # new_loss = None
            
            # while not found_step and line_search_iters < max_line_search_iters:
            #     # Calculate new candidate solution
            #     new_sol = sol + alpha * newton_direction
            #     new_loss = self.loss_function(new_sol, rhs_f, bdy_g, L)
                
            #     # Check Armijo condition
            #     sufficient_decrease = new_loss <= current_loss + c * alpha * directional_derivative
                
            #     if sufficient_decrease:
            #         found_step = True
            #     else:
            #         # Reduce step size
            #         alpha *= beta
            #         line_search_iters += 1
            
            # Update solution
            sol = sol + alpha * newton_direction
            
            # Calculate current loss
            J_now = self.loss_function(sol, rhs_f, bdy_g, L)
            J_hist.append(J_now)
            
            # # If loss decreases, reduce regularization parameter
            # if J_now < current_loss:
            #     damping = max(damping * damping_decrease, initial_damping)
            # else:
            #     # If loss doesn't decrease, increase regularization parameter
            #     damping = min(damping * damping_factor, max_damping)
            
            # Print update details
            if iter_step % 10 == 0 or iter_step == optimizer_steps - 1:
                print(f"Iteration {iter_step}: Loss = {J_now}, Gradient norm = {grad_norm}, Damping = {damping}, Step size = {alpha}")
        
        self.loss_history = J_hist
        
        # Calculate final feature vector z
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
    
        return phi_x.astype(jnp.float16)
    
    def predict(self, x_t_infer):
        '''Predict the solution at x_t_infer using a scalar function'''
        right_vector = self.right_vector.squeeze()  # Shape: (col_dim,)
    
        # Define a function representing the solution at a single x_t point
        def solution_function(x_t):
            # Compute kernel vector between x_t and phi (training data)
            kernel_values = self.kernel_x_t_phi_single(x_t)  # Shape: (col_dim,)
            # Compute the scalar prediction
            sol = jnp.dot(kernel_values, right_vector)
            return sol.astype(jnp.float16)
    
        # Vectorize the solution function over x_t_infer
        solution_function_vectorized = jit(vmap(solution_function))
    
        # Compute the predictions
        sol_infer = solution_function_vectorized(x_t_infer)  # Shape: (N_infer,)
    
        return sol_infer[:,jnp.newaxis].astype(jnp.float16)
    
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
    
        return gradient.astype(jnp.float16)
    
    def compute_PDE_loss(self,x_t_infer):
        '''Compute the PDE loss at x_t_infer'''
        raise NotImplementedError

class GP_Grad_Dependent_Nonlinear(GP):
    '''Gaussian Kernel Solver for the Grad_Dependent_Nonlinear'''

    def __init__(self, equation):
        super(GP_Grad_Dependent_Nonlinear, self).__init__(equation)

    
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

        return F_domain.astype(jnp.float16)

    
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


        return DF_domain_without_time.astype(jnp.float16)

    
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
        return loss.astype(jnp.float16)



    
    

    

    
    
