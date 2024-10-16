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
F(u,z)(x,t)=2
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
y_{domain}=-2-\frac{\partial u}{\partial t}+\frac1d div_x u=-2
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

Feature vector $\phi$ have $M=M_{\Omega}+ M_{\partial \Omega}+ M_{\Omega}$ components, which is also the size of the kernel matrix.

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
d=100, \mu=-1/d,\sigma=\sqrt{2}, D=[-0.5,0.5]^{100}, s=0, T=0.5
$$

and

$$
d=250,\mu=-1/d, \sigma=\sqrt{2}, D=[-0.5,0.5]^{250}, s=0, T=0.5
$$

