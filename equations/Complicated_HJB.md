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
We define
$$
\vec{y}_{domain}=-2\cdot 1_{M_\Omega}\\
\vec{y}_{boundary}=g((x,t)_{1:M_{\partial \Omega}}^{\partial \Omega})
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
\mathbf{y}=\begin{pmatrix}\vec{y}_{domain}\\\vec{y}_{boundary}\end{pmatrix}
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

