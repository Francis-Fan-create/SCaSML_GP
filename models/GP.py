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
        phi_dim = 2* N_domain+N_boundary
        kernel_phi_phi = np.zeros((phi_dim, phi_dim))
        for i in range(N_domain):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.kappa(x_t_domain[i], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.kappa(x_t_domain[i], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, phi_dim):
                kernel_phi_phi[i,j] = self.laplacian_y_t_kappa(x_t_domain[i], x_t_domain[j-N_domain-N_boundary])
        for i in range(N_domain, N_domain+N_boundary):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.kappa(x_t_boundary[i-N_domain], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.kappa(x_t_boundary[i-N_domain], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, phi_dim):
                kernel_phi_phi[i,j] = self.laplacian_y_t_kappa(x_t_boundary[i-N_domain], x_t_domain[j-N_domain-N_boundary])
        for i in range(N_domain+N_boundary, phi_dim):
            for j in range(N_domain):
                kernel_phi_phi[i, j] = self.laplacian_x_t_kappa(x_t_domain[i-N_domain-N_boundary], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_phi_phi[i, j] = self.laplacian_x_t_kappa(x_t_domain[i-N_domain-N_boundary], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, phi_dim):
                kernel_phi_phi[i,j] = self.laplacian_x_t_laplacian_y_t_kappa(x_t_domain[i-N_domain-N_boundary], x_t_domain[j-N_domain-N_boundary])
        return kernel_phi_phi
    
    def kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the kernel matrix K(x_t,phi)'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        row_dim = N_infer
        col_dim = 2* N_domain+N_boundary
        '''Compute the kernel matrix K(x_t,phi)'''
        kernel_x_phi = np.zeros((row_dim, col_dim))
        for i in range(N_infer):
            for j in range(N_domain):
                kernel_x_phi[i, j] = self.kappa(x_t_infer[i], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                kernel_x_phi[i, j] = self.kappa(x_t_infer[i], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, col_dim):
                kernel_x_phi[i,j] = self.laplacian_y_t_kappa(x_t_infer[i], x_t_domain[j-N_domain-N_boundary])
        return kernel_x_phi
    
    def dx_t_kernel_x_t_phi(self, x_t_infer, x_t_domain, x_t_boundary):
        '''Compute the gradient of the kernel matrix K(x_t,phi) with respect to x_t'''
        N_infer = x_t_infer.shape[0]
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        row_dim = N_infer
        col_dim = 2* N_domain+N_boundary
        dx_t_kernel_x_phi = np.zeros((row_dim, col_dim, self.n_input))
        for i in range(N_infer):
            for j in range(N_domain):
                dx_t_kernel_x_phi[i, j] = self.dx_t_kappa(x_t_infer[i], x_t_domain[j])
            for j in range(N_domain, N_domain+N_boundary):
                dx_t_kernel_x_phi[i, j] = self.dx_t_kappa(x_t_infer[i], x_t_boundary[j-N_domain])
            for j in range(N_domain+N_boundary, col_dim):
                dx_t_kernel_x_phi[i,j] = self.laplacian_y_t_kappa(x_t_infer[i], x_t_domain[j-N_domain-N_boundary])* -(x_t_infer[i]-x_t_domain[j-N_domain-N_boundary])/self.sigma**2
        return dx_t_kernel_x_phi
    
    def y_domain(self,x_t):
        '''Compute the nonlinear term on the right at x_t'''
        raise NotImplementedError
    
    def y_boundary(self,x_t):
        '''Compute the terminal condition at x_t'''
        return self.equation.g(x_t)[:,np.newaxis]
    

    '''The following part is a version based on sparse Cholesky decomposion, which is not used in the final version due to the numerical instability'''
    # def maximin_ordering(self, points, A=None):
    #     """
    #     Perform maximin ordering on the given points conditioned on set A.
    #     """
    #     n = len(points)
    #     ordered_points = np.zeros_like(points)
    #     remaining_points = points.copy()
    #     P_perm = np.eye(n, dtype=int)
    #     lengthscales = np.zeros(n)
        
    #     # Start with the first point
    #     if A is None:
    #         ordered_points[0] = remaining_points[0]
    #         remaining_points = np.delete(remaining_points, 0, axis=0)
    #         lengthscales[0] = np.inf
    #     else:
    #         distances = cdist(A, remaining_points)
    #         maximin_index = np.argmax(np.min(distances, axis=0))
    #         ordered_points[0] = remaining_points[maximin_index]
    #         P_perm[:, [0, maximin_index]] = P_perm[:, [maximin_index, 0]]
    #         remaining_points = np.delete(remaining_points, maximin_index, axis=0)
    #         lengthscales[0] = np.min(distances[:, maximin_index])
        
    #     for i in range(1, n):
    #         if A is None:
    #             distances = cdist(ordered_points[:i], remaining_points)
    #             min_distances = np.min(distances, axis=0)
    #             maximin_index = np.argmax(min_distances)
    #             ordered_points[i] = remaining_points[maximin_index]
    #             P_perm[:, [i, maximin_index]] = P_perm[:, [maximin_index, i]]
    #             remaining_points = np.delete(remaining_points, maximin_index, axis=0)
    #             lengthscales[i] = min_distances[maximin_index]
    #         else:
    #             dist_eval_points = np.concatenate((A, ordered_points[:i]), axis=0)
    #             distances = cdist(dist_eval_points, remaining_points)
    #             min_distances = np.min(distances, axis=0)
    #             maximin_index = np.argmax(min_distances)
    #             ordered_points[i] = remaining_points[maximin_index]
    #             P_perm[:, [i, maximin_index]] = P_perm[:, [maximin_index, i]]
    #             remaining_points = np.delete(remaining_points, maximin_index, axis=0)
    #             lengthscales[i] = min_distances[maximin_index]
        
    #     return ordered_points, P_perm, lengthscales

    # def construct_sparsity_pattern(self, ordered_points, lengthscales, rho, lambda_param):
    #     """
    #     Construct the aggregate sparsity pattern based on the given parameters using supernode method.
    #     """
    #     n = len(ordered_points)
    #     sparsity_pattern = np.zeros((n, n))
    #     aggregated = np.zeros(n, dtype=bool)
    #     supernodes = []

    #     for j in range(n-1, -1, -1):
    #         if not aggregated[j]:
    #             supernode = [j]
    #             aggregated[j] = True
    #             for i in range(j):
    #                 if not aggregated[i] and np.linalg.norm(ordered_points[i] - ordered_points[j]) <= rho * lengthscales[j] and lengthscales[i] <= lambda_param * lengthscales[j]:
    #                     supernode.append(i)
    #                     aggregated[i] = True
    #             supernodes.append(supernode)

    #     for supernode in supernodes:
    #         for i in supernode:
    #             for j in supernode:
    #                 sparsity_pattern[i, j] = 1
    #                 sparsity_pattern[j, i] = 1

    #     return sparsity_pattern
    
    # def kl_minimization(self, K, sparsity_pattern):
    #     """
    #     Perform KL minimization to obtain the sparse Cholesky factor.
    #     """
    #     n = K.shape[0]
    #     U = np.zeros_like(K)
        
    #     for j in range(n):
    #         s_j = np.where(sparsity_pattern[:, j] == 1)[0]
    #         if len(s_j) == 0:
    #             continue
    #         Theta_sj_sj = K[np.ix_(s_j, s_j)]
    #         e_sj = np.zeros(len(s_j))
    #         e_sj[-1] = 1
    #         U_sj_j = solve_triangular(Theta_sj_sj, e_sj, lower=True)
    #         U[s_j, j] = U_sj_j / np.sqrt(e_sj.T @ U_sj_j)
        
    #     return U

    # def sparse_cholesky_factorization(self, phi, rho, lambda_):
    #     """
    #     Perform sparse Cholesky factorization for K(phi, phi)^-1.
        
    #     Args:
    #         phi (np.ndarray): Measurements, shape (n_samples, n_features).
    #         kernel_function (callable): Kernel function K.
    #         rho (float): Sparsity parameter.
    #         lambda_ (float): Supernodes parameter.
        
    #     Returns:
    #         np.ndarray: Sparse Cholesky factor U_rho.
    #         np.ndarray: Permutation matrix P_perm.
    #     """
    #     # Step 3: Reordering and sparsity pattern
    #     interior_points = phi[:self.N_domain]
    #     boundary_points = phi[self.N_domain:]
        
    #     ordered_boundary_points, P_perm_boundary, lengthscales_boundary = self.maximin_ordering(boundary_points)
    #     ordered_interior_points, P_perm_interior, lengthscales_interior = self.maximin_ordering(interior_points, A=ordered_boundary_points)
        
    #     ordered_points = np.vstack((ordered_interior_points, ordered_boundary_points))
    #     P_perm = np.block([
    #         [P_perm_interior, np.zeros((P_perm_interior.shape[0], P_perm_boundary.shape[1]), dtype=int)],
    #         [np.zeros((P_perm_boundary.shape[0], P_perm_interior.shape[1]), dtype=int), P_perm_boundary]
    #     ])
        
    #     lengthscales = np.concatenate((lengthscales_interior, lengthscales_boundary))
    #     sparsity_pattern = self.construct_sparsity_pattern(ordered_points, lengthscales, rho, lambda_)
        
    #     # Step 4: KL minimization
    #     K = self.kappa_kernel(ordered_points, ordered_points)
    #     U_rho = self.kl_minimization(K, sparsity_pattern)
        
    #     return U_rho, P_perm

    # def pCG(self, A, b, M, tol=1e-6, max_iter=1000):
    #     """
    #     Preconditioned Conjugate Gradient method to solve Ax = b with preconditioner M.
    #     """
    #     x = np.zeros((self.N_domain + self.N_boundary, 1))
    #     r = b - A @ x
    #     z = np.linalg.solve(M, r)
    #     p = z.copy()
    #     rsold = r.T @ z
    
    #     for i in range(max_iter):
    #         Ap = A @ p
    #         pAp = p.T @ Ap
    
    #         if pAp == 0:
    #             pAp += 1e-10
    
    #         alpha = rsold / pAp
    
    #         alpha = np.clip(alpha, -1e10, 1e10)
    
    #         x += alpha * p
    #         r -= alpha * Ap
    
    #         r = np.clip(r, -1e10, 1e10)
    
    #         if np.linalg.norm(r) < tol:
    #             break
    
    #         z = np.linalg.solve(M, r)
    #         rsnew = r.T @ z
    
    #         if rsold == 0:
    #             rsold += 1e-10
    
    #         p = z + (rsnew / rsold) * p
    #         rsold = rsnew
    
    #     return x

    # def sparse_cholesky_gauss_newton(self, phi, F, DF, y, kernel_function, t, rho, rho_r, lambda_):
    #     """
    #     Sparse Cholesky accelerated Gauss-Newton for solving (2.5).
        
    #     Args:
    #         phi (np.ndarray): Measurements, shape (n_samples, n_features).
    #         F (callable): Operator F
    #         DF (callable): Derivative of the operator F.
    #         y (np.ndarray): Data vector.
    #         kernel_function (np.ndarray): Kernel function K.
    #         t (int): Number of Gauss-Newton steps.
    #         rho (float): Sparsity parameter.
    #         rho_r (float): Sparsity parameter for reduced measurements.
    #         lambda_ (float): Supernodes parameter.
        
    #     Returns:
    #         np.ndarray: Solution z_t.
    #     """
    #     # Step 3: Factorize K(phi, phi)^-1
    #     U_rho, P_perm = self.sparse_cholesky_factorization(phi, rho, lambda_)
        
    #     # Step 4: Initialize
    #     k = 0
    #     z_k = np.random.uniform(-1,1,(2*self.N_domain+self.N_boundary,1))
        
    #     while k < t:
    #         # Step 6: Form the reduced measurements
    #         DF_zk = DF(z_k)
    #         phi_k = DF_zk @ phi
            
    #         # Step 7: Factorize K(phi_k, phi_k)^-1
    #         A = DF_zk @ kernel_function @ DF_zk.T
    #         U_rho_r, Q_perm_r = self.sparse_cholesky_factorization(phi_k, rho_r, lambda_)
            
    #         # Step 8: Use pCG to solve (5.4)
    #         b = y - F(z_k)+ DF_zk @ z_k
    #         M = Q_perm_r.T @ U_rho_r @ U_rho_r.T @ Q_perm_r + self.nugget * np.eye(self.N_domain+self.N_boundary)
    #         gamma = self.pCG(A, b, M)
            
    #         # Step 9: Update z_k
    #         z_k = np.linalg.solve(P_perm.T @ U_rho @ U_rho.T @ P_perm + self.nugget*np.eye(2*self.N_domain+self.N_boundary), DF_zk.T @ gamma)
            
    #         k += 1
            
    #     self.right_vector = np.linalg.solve(kernel_function + self.nugget*np.eye(2*self.N_domain+self.N_boundary), z_k)
        
    #     return z_k 


    
    # def GPsolver(self, x_t_domain, x_t_boundary, GN_step=4):
    #     """
    #     Solve the Gaussian Process for the given domain and boundary points.

    #     Args:
    #         x_t_domain (np.ndarray): Coordinates of the domain points, shape (N_domain, n_input).
    #         x_t_boundary (np.ndarray): Coordinates of the boundary points, shape (N_boundary, n_input).
    #         GN_step (int, optional): Number of Gauss-Newton iterations. Default is 4.

    #     Returns:
    #         np.ndarray: Solution after Gauss-Newton iterations on x_t_domain, shape (N_domain,1).
    #     """
    #     N_domain = x_t_domain.shape[0]
    #     N_boundary = x_t_boundary.shape[0]
    #     self.x_t_domain = x_t_domain
    #     self.x_t_boundary = x_t_boundary
    #     self.N_domain = N_domain
    #     self.N_boundary = N_boundary
    #     z = np.random.uniform(-1,1,(2* N_domain+N_boundary,1))
    #     y = np.concatenate((self.y_domain(x_t_domain),self.y_boundary(x_t_boundary)), axis=0)
    #     kernel_phi_phi = self.kernel_phi_phi(x_t_domain, x_t_boundary)  # Kernel matrix between z and z
    #     z = self.sparse_cholesky_gauss_newton(z, self.F, self.DF, y, kernel_phi_phi, GN_step, 10, 10, 1.2)
    #     return z[:N_domain]

    def GPsolver(self, x_t_domain, x_t_boundary, GN_step=4):
        '''Solve the Gaussian Process for the given domain and boundary points'''
        N_domain = x_t_domain.shape[0]
        N_boundary = x_t_boundary.shape[0]
        self.x_t_domain = x_t_domain
        self.x_t_boundary = x_t_boundary
        self.N_domain = N_domain
        self.N_boundary = N_boundary
        z_k = np.random.uniform(-1,1,(2* N_domain+N_boundary,1))
        y = np.concatenate((self.y_domain(x_t_domain),self.y_boundary(x_t_boundary)), axis=0)
        gamma = np.zeros((N_domain+N_boundary,1))
        kernel_phi_phi = self.kernel_phi_phi(x_t_domain, x_t_boundary) # Kernel matrix between z and z
        for i in range(GN_step):
            DF_k = self.DF(z_k)
            proj_kernel_phi_phi = DF_k @ kernel_phi_phi @ DF_k.T
            gamma = np.linalg.solve(proj_kernel_phi_phi+self.nugget*np.eye(N_domain+N_boundary), (y-self.F(z_k)+DF_k @ z_k))
            z_k_plus_1 = kernel_phi_phi @ DF_k.T @ gamma
            z_k = z_k_plus_1
        self.right_vector = np.linalg.solve(kernel_phi_phi+self.nugget*np.eye(2* N_domain+N_boundary), z_k)
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
        F = np.zeros((N_domain+N_boundary,1))
        F[:N_domain] = z[N_domain+N_boundary:]
        F[N_domain:] = z[N_domain:N_domain+N_boundary]
        return F
    
    def DF(self,z):
        '''Compute the jacobian of the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        DF = np.zeros((N_domain+N_boundary,2*N_domain+N_boundary))
        for i in range(N_domain):
            DF[i,:N_domain] = 0
            DF[i,N_domain:2*N_domain] = 0
            DF[i,2*N_domain:] = 1
        for i in range(N_domain,N_domain+N_boundary):
            DF[i,:N_domain] = 0
            DF[i,N_domain:2*N_domain] = 1
            DF[i,2*N_domain:] = 0
        return DF

class GP_Explicit_Solution_Example(GP):
    '''Gaussian Kernel Solver for the Explicit Solution Example'''
    def __init__(self,equation):
        super(GP_Explicit_Solution_Example, self).__init__(equation)

    def y_domain(self,x_t):
        sigma = self.equation.sigma()
        d= self.d
        t = x_t[:,-1]
        result= (-2/sigma**2+1+2/(d*sigma**2)-(2*np.exp(t+np.sum(x_t[:,:-1],axis=1)))/(1+np.exp(t+np.sum(x_t[:,:-1],axis=1))))*(np.exp(t+np.sum(x_t[:,:-1],axis=1)))/((1+np.exp(t+np.sum(x_t[:,:-1],axis=1)))**2)
        return result[:,np.newaxis]
    
    
    def F(self,z):
        '''Compute the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        d = self.d
        F = np.zeros((N_domain+N_boundary,1))
        F[:N_domain] = z[N_domain+N_boundary:]
        F[N_domain:] = z[N_domain:N_domain+N_boundary]
        return F
    
    def DF(self,z):
        '''Compute the jacobian of the operator F at z'''
        N_domain = self.N_domain
        N_boundary = self.N_boundary
        DF = np.zeros((N_domain+N_boundary,2*N_domain+N_boundary))
        for i in range(N_domain):
            DF[i,:N_domain] = 0
            DF[i,N_domain:2*N_domain] = 0
            DF[i,2*N_domain:] = 1
        for i in range(N_domain,N_domain+N_boundary):
            DF[i,:N_domain] = 0
            DF[i,N_domain:2*N_domain] = 1
            DF[i,2*N_domain:] = 0
        return DF
    

    
    
