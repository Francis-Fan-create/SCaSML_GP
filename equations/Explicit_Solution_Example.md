# Explicit_Solution_Example

## MLP Setting

We solve the following [PDE](https://arxiv.org/abs/1708.03223):
$$
\frac{\partial u}{\partial t}+(\sigma^2u(x,t)-\frac 1d -\frac{\sigma^2}{2})div_x u(x,t)+\frac {\sigma^2}2 \Delta_x u(x,t)=0,t\in[s,T],x\in D \in \mathbb{R}^d
$$

whose terminal condition is given by

$$
u(x,T)=g(x):=\frac{\exp(T+\sum\limits_{i=1}^d x_i)}{1+\exp(T+\sum\limits_{i=1}^d x_i)}
$$

without any boundary constraints.



Then nonlinear term is given by
$$
F(u,z)(x,t)=(\sigma u-\frac {1}{d\sigma} -\frac{\sigma}{2})\sum_i z
$$


This PDE has an explicit solution at time $t$:
$$
u(x,t)=\frac{\exp(t+\sum\limits_{i=1}^d x_i)}{1+\exp(t+\sum\limits_{i=1}^d x_i)}
$$

which is our target in this section.

## GP Setting

Rewrite the PDE as
$$
\frac{\partial u}{\partial t}+(\sigma^2u(x,t)-\frac 1d -\frac{\sigma^2}{2})div_x u(x,t)+\frac {\sigma^2}2 \Delta_x u(x,t)=0,t\in[s,T],x\in D \in \mathbb{R}^d
$$
We define
$$
\vec{y}_{domain}=0_{M_{\Omega}}\\
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
\phi^1_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ u ,1\leq j\leq M_\Omega\\
\phi^2_j(u)&:u\rightarrow \delta_{x_{\partial\Omega}^j}\circ u,1\leq j\leq M_{\partial \Omega}\\
\phi^3_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ L_1 \circ u,1\leq j\leq M_\Omega\\
\phi^4_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ L_2 \circ u,1\leq j\leq M_\Omega\\
\phi^5_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ L_3 \circ u,1\leq j\leq M_\Omega\\
\end{align}
$$
Denote $\vec{z}_i=\phi^i_{1:M_\Omega}(u),i\in\{1,3,4,5\}$ and $\vec{z}_i=\phi^i_{1:M_{\partial \Omega}}(u),i=2$.

Let 
$$
\vec{z}=\begin{pmatrix}\vec{z}_1\\\vec{z}_2\\\vec{z}_3\\\vec{z}_4\\\vec{z}_5\end{pmatrix}
$$
We then derive
$$
\begin{align}
F(\vec{z})=\begin{pmatrix}\vec{z}_4+\sigma^2\vec{z}_1\odot\vec{z}_5-(\frac1d +\frac{\sigma^2}{2})\vec{z}_5+\frac{\sigma^2}{2}\vec{z}_3\\\vec{z}_2\end{pmatrix}
\end{align}
$$
Thus $Q_\Omega=4, Q=5$. 

Feature vector $\phi$ have $M=M_{\partial \Omega}+ 4M_{\Omega}$ components, which is also the size of the kernel matrix.

Taking derivatives under linearization condition
$$
\begin{align*}
DF(\vec{z}_{k})=\begin{pmatrix}\sigma^2\text{diag}(\vec{z}_5) && 0 && \frac{\sigma^2}{2}I_{M_{\Omega}}&& I_{M_{\Omega}} && \sigma^2\text{diag}(\vec{z}_1)-(\frac1d+\frac{\sigma^2}{2})I_{M_{\Omega}} \\
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
d=100, \mu=0, \sigma=0.25, D=[-0.5,0.5]^{100}, s=0, T=0.5
$$

and

$$
d=250, \mu=0, \sigma=0.25, D=[-0.5,0.5]^{250}, s=0, T=0.5
$$