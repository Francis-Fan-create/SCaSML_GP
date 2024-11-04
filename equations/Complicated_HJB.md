# Complicated_HJB

## MLP Setting

We solve the following [PDE](https://arxiv.org/abs/2206.02016)(labeled as Eq(100) in the original paper)
$$
\frac{\partial u}{\partial t}-\frac1d div_x u(x,t)+2+\Delta_x u(x,t)=0,t\in[s,T],x\in D⊂\mathbb{R}^d
$$

whose terminal condition is given by


$$
u(x,T)=g(x):=\sum_{i=1}^d x_i,
$$

without any boundary constraints.

The nonlinear term is given by
$$
F(u,z)(x,t)=\sqrt{2}
$$

This PDE has an explicit solution at time $t$
$$
u(x,t)=\sum_{i=1}^d x_i+(T-t).
$$

which is our target in this section.

## GP Setting

Rewrite the PDE as
$$
\frac{\partial u}{\partial t}-\frac1d div_x u(x,t)+\Delta_x u(x,t)=-2,t\in[s,T],x\in D⊂\mathbb{R}^d
$$
Let the nonlinear term be
$$
\vec{y}_{domain}=-2\cdot 1_{M_\Omega}
$$
Define operators
$$
\begin{align}
L_1(u):u&\rightarrow \Delta_x u\\
L_2(u):u&\rightarrow \frac{\partial u}{\partial t}\\
L_3(u):u&\rightarrow div_x u
\end{align}
$$
We define feature functions
$$
\begin{align}
\phi^1_j(u)&:u\rightarrow \delta_{(x,t)_\Omega^j}\circ u ,1\leq j\leq M_\Omega\\
\phi^2_j(u)&:u\rightarrow \delta_{(x,t)_{\partial\Omega}^j}\circ u,1\leq j\leq M_{\partial \Omega}\\
\phi^3_j(u)&:u\rightarrow \delta_{(x,t)_\Omega^j}\circ L_1 \circ u,1\leq j\leq M_\Omega\\
\phi^4_j(u)&:u\rightarrow \delta_{(x,t)_\Omega^j}\circ L_2 \circ u,1\leq j\leq M_\Omega\\
\phi^5_j(u)&:u\rightarrow \delta_{(x,t)_\Omega^j}\circ L_3 \circ u,1\leq j\leq M_\Omega\\
\end{align}
$$
We note that $u=\sum_{i=1}^5 \vec{w}_i \cdot\vec\phi^i$, where $\vec{w}_i$ is the corresponding coefficient of feature $\vec\phi^i$.

Denote $\vec{z}_i=\phi^i_{1:M_\Omega}(u),i\in\{1,3,4,5\}$ and $\vec{z}_i=\phi^i_{1:M_{\partial \Omega}}(u),i=2$.

Concatenate the features
$$
\vec{z}=\begin{pmatrix}\vec{z}_1\\\vec{z}_2\\\vec{z}_3\\\vec{z}_4\\\vec{z}_5\end{pmatrix}
$$
and these coefficients
$$
\vec{w}=\begin{pmatrix}\vec{w}_1\\\vec{w}_2\\\vec{w}_3\\\vec{w}_4\\\vec{w}_5\end{pmatrix}
$$
We have: $u=\vec{w}\cdot\vec{z}$.

We then define
$$
\begin{align}
F(\vec{z})=\begin{pmatrix}\vec{z}_4-\frac1d \vec{z}_5+\vec{z}_3\\\vec{z}_2\end{pmatrix}
\end{align}
$$
Feature vector $\phi$ has $M=4M_{\partial \Omega}+ M_{\Omega}$ components, which is also the size of the kernel matrix.

Taking derivatives
$$
\begin{align*}
DF(\vec{z}_{k})=\begin{pmatrix}0 && 0 && I_{M_{\Omega}} && I_{M_{\Omega}} && -\frac1d I_{M_{\Omega}}\\
0 && I_{M_{\partial \Omega}} && 0 && 0 && 0\end{pmatrix}
\end{align*}
$$
The problem to solve is
$$
\min \vec{z}^T K(\phi,\phi)^{-1}\vec{z}\\
s.t. F(\vec{z})=\mathbf{y}
$$
where
$$
\mathbf{y}=\begin{pmatrix}\vec{y}_{domain}\\g((x,t)_{1:M_{\partial \Omega}}^{\partial \Omega})\end{pmatrix}
$$
We solve the problem as the following:

Step 1: Construct $\mathbf{y}$ based on the nonlinear term
$$
\mathbf{y}=\begin{pmatrix}\vec{y}_{domain}\\g((x,t)_{1:M_{\partial \Omega}}^{\partial \Omega})\end{pmatrix}
$$
Step 2: We solve the system 
$$
F(\vec{z})=DF(\vec{z})\vec{z}=\vec{y}.
$$
This derives
$$
\vec{z} =\vec{z}_s +Z_0 \cdot \vec{w}
$$
where $z_s$ is a given solution, $Z_0$ is the basis of the null space of $DF$, and $\vec{w}$ are parameters to optimize.

Step 3: We consider the following optimization problem using the parameterization in step 2:
$$
\min_{\vec{w}} \vec{z}^T K(\phi,\phi)^{-1} \vec{z}
$$
where:
$$
K(\phi,\phi)=\\\begin{pmatrix}
K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) && \Delta_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& D_{t_y}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&div_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})\\
K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) && \Delta_y K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& D_{t_y}K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&div_y K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})\\
\Delta_x K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& \Delta_x K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) && \Delta_x\Delta_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&\Delta_x D_{t_y}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&\Delta_x div_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})\\
D_{t_x}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&D_{t_x} K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) &&D_{t_x} \Delta_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& D_{t_x}D_{t_y}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&D_{t_x} div_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})\\
div_x K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&div_x  K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) &&div_x  \Delta_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& div_x  D_{t_y}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&div_x div_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})
\end{pmatrix}
$$
Under the RBF kernel we use, for $\tilde{\sigma}=\sigma\cdot\sqrt{d}$, we have:
$$
K(\phi,\phi) = \\\begin{pmatrix}
e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{t_y-t_x}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{\sum(y-x)}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} \\

e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{t_y-t_x}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{\sum(y-x)}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} \\

(-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (\frac{d(d+2)}{\tilde{\sigma}^4}-\frac{2(d+2)\|x-y\|^2}{\tilde{\sigma}^6}+\frac{\|x-y\|^4}{\tilde{\sigma}^8})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & \frac{t_x-t_y}{\tilde{\sigma}^2}(-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (\frac{(d+1)\sum(y-x)}{\tilde{\sigma}^4}+\frac{\sum(y-x)}{\tilde{\sigma}^6}-\frac{\|y-x\|^2\sum(y-x)}{\tilde{\sigma}^8})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} \\

-\frac{t_x-t_y}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{t_x-t_y}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & \frac{t_y-t_x}{\tilde{\sigma}^2}(-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{(t_x-t_y)^2}{\tilde{\sigma}^4}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & \frac{t_y-t_x}{\tilde{\sigma}^2}(-\frac{\sum(x-y)}{\tilde{\sigma}^2})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} \\

-\frac{\sum(x-y)}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{\sum(x-y)}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (\frac{(d+1)\sum(x-y)}{\tilde{\sigma}^4}+\frac{\sum(x-y)}{\tilde{\sigma}^6}-\frac{\|x-y\|^2\sum(x-y)}{\tilde{\sigma}^8})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{(t_x-t_y)\sum(x-y)}{\tilde{\sigma}^4}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (\frac{d}{\tilde{\sigma}^2}-\frac{(\sum(x-y))^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}}
\end{pmatrix}
$$
Step 4: Compute $\vec{z}^\star$
$$
\vec{z}^{\star}=\vec{z}_s - Z_0\cdot (Z_0^TK(\phi,\phi)^{-1}Z_0)^{-1}(Z_0^TK(\phi,\phi)^{-1}\vec{z}_s)
$$
by directly solving $\vec{w}^{\star}$ from the quadratic problem in step 3

Step 5: Return the solution
$$
u((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}})=K((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}},\phi)K(\phi,\phi)^{-1}\vec{z}^{\star}
$$
where we have
$$
K((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}},\phi)=\begin{pmatrix}
K((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}},\phi),(y,t_y)_\Omega^{1:M_{\Omega}})&& K((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}},\phi),\phi),(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) && \Delta_y K((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}},\phi),(y,t_y)_\Omega^{1:M_{\Omega}})&& D_{t_y}K((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}},\phi),(y,t_y)_\Omega^{1:M_{\Omega}})&&div_y K((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}},\phi),(y,t_y)_\Omega^{1:M_{\Omega}})
\end{pmatrix}
$$
i.e.
$$
K((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}},\phi)=\begin{pmatrix}
e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{t_y-t_x}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{\sum(y-x)}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} \end{pmatrix}
$$


## Parameters

Specifically, we consider the problem for
$$
d=100, \mu=-1/d,\sigma=\sqrt{2}, D=[-0.5,0.5]^{100}, s=0, T=0.5
$$

and

$$
d=250,\mu=-1/d, \sigma=\sqrt{2}, D=[-0.5,0.5]^{250}, s=0, T=0.5
$$

