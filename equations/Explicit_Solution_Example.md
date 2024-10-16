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
Let the nonlinear term be
$$
y_{domain}=-\frac{2}{\sigma^2}\frac{\partial u}{\partial t}+(1+\frac{2}{d\sigma^2}-2u)div_x u=(-\frac{2}{\sigma^2}+1+\frac{2}{d\sigma^2}-\frac{2\exp(t+\sum_{i=1}^d x_i)}{1+\exp(t+\sum_{i=1}^d x_i)})\frac{\exp(t+\sum_{i=1}^d x_i)}{(1+\exp(t+\sum_{i=1}^d x_i))^2}
$$
Define operators
$$
\begin{align}
L(u):u&\rightarrow \Delta_x u
\end{align}
$$
We define feature functions
$$
\begin{align}
\phi^1_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ u ,1\leq j\leq M_\Omega\\
\phi^2_j(u)&:u\rightarrow \delta_{x_{\partial\Omega}^j}\circ u,1\leq j\leq M_{\partial \Omega}\\
\phi^3_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ L \circ u,1\leq j\leq M_\Omega\\
\end{align}
$$
Denote $\vec{z}_i=\phi^i_{1:M_\Omega}(u),i\in\{1,3\}$ and $\vec{z}_i=\phi^i_{1:M_{\partial \Omega}}(u),i=2$.

Let 
$$
\vec{z}=\begin{pmatrix}\vec{z}_1\\\vec{z}_2\\\vec{z}_3\end{pmatrix}
$$
We then derive
$$
\begin{align}
F(\vec{z})=\begin{pmatrix}\vec{z}_3\\\vec{z}_2\end{pmatrix}
\end{align}
$$
Thus $Q_\Omega=2, Q=3$. 

Feature vector $\phi$ have $M=M_{\Omega} +M_{\partial \Omega}+ M_{\Omega}$ components, which is also the size of the kernel matrix.

Taking derivatives under linearization condition
$$
\begin{align*}
DF(\vec{z}_{k})=\begin{pmatrix}0 && 0 && I_{M_{\Omega}}\\
0 && I_{M_{\partial \Omega}} && 0\end{pmatrix}
\end{align*}
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