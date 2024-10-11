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
f(x,t)=-2
$$
Define operators
$$
\begin{align}
L_1(u):u&\rightarrow\frac{\partial u}{\partial t}\\
L_2(u):u&\rightarrow div_x u\\
L_3(u):u&\rightarrow \Delta_x u
\end{align}
$$
We define feature functions
$$
\begin{align}
\phi^1_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ u ,1\leq j\leq M_\Omega\\
\phi^2_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ L_1 \circ u,1\leq j\leq M_\Omega\\
\phi^3_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ L_2 \circ u,1\leq j\leq M_\Omega\\
\phi^4_j(u)&:u\rightarrow \delta_{x_\Omega^j}\circ L_3 \circ u,1\leq j\leq M_\Omega\\
\phi^5_j(u)&:u\rightarrow \delta_{x_{\partial\Omega}^j}\circ u,1\leq j\leq M_{\partial \Omega}\\
\end{align}
$$
Denote $\vec{z}_i=\phi^i_{1:M_\Omega}(u),1\leq i\leq 4$ and $\vec{z}_i=\phi^i_{1:M_{\partial \Omega}}(u),i=5$.

We then derive
$$
\begin{align}
P(\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)&=\vec{z}_2-\frac{1}{d}\vec{z}_3+\vec{z}_4\\
B(\vec{z}_5)&=\vec{z}_5
\end{align}
$$
Thus $Q_\Omega=4, Q=5$. 

Feature vector $\phi$ have $N=M_\Omega+M_\Omega+M_\Omega+M_\Omega+(M-M_\Omega)$ components, which is also the size of the kernel matrix.

Taking derivatives under linearization condition
$$
\begin{align*}
\nabla_z P(\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)&=\begin{pmatrix}0_{M_\Omega}\\1_{M_\Omega}\\-\frac{1}{d}\cdot1_{M_\Omega}\\1_{M_\Omega}\end{pmatrix}\\
\nabla_z B(\vec{z}_5)&=1_{M_{\partial \Omega}}
\end{align*}
$$
Concatenate $\vec{z}_i,1\leq i\leq 5$:
$$
\vec{z}=\begin{pmatrix}\vec{z}_1\\\vec{z}_2\\\vec{z}_3\\\vec{z}_4\\\vec{z}_5\\\end{pmatrix}
$$
The iteration step becomes
$$
\begin{align}
K(\phi_l,\phi_l)\vec{\gamma}_l&=\begin{pmatrix}f(x_{1:M_\Omega})-P((\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)_l)\\f(x_{1:M_\Omega})-P((\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)_l)\\f(x_{1:M_\Omega})-P((\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)_l)\\f(x_{1:M_\Omega})-P((\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)_l)\\g(x_{1:M_{\partial\Omega}})-B((\vec{z}_5)_l)\end{pmatrix}+\begin{pmatrix}\nabla_z P(\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)\odot\begin{pmatrix}\vec{z}_1\\\vec{z}_2\\\vec{z}_3\\\vec{z}_4\end{pmatrix}_l\\\nabla_z B(\vec{z}_5)\odot(\vec{z}_5)_l\end{pmatrix}\\
\vec{z}_{l+1}&=K(\phi_l,\phi_l)\begin{pmatrix}\nabla_z P(\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)\\\nabla_z B(\vec{z}_5)\end{pmatrix}^T \vec{\gamma}_l
\end{align}
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

