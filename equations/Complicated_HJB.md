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
\nabla_z P(\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)&=(0_{M_\Omega},1_{M_\Omega},-1_{M_\Omega}/d,1_{M_\Omega})&&\Rightarrow \frac{d}{du}P(u)=(0_{M_\Omega},1_{M_\Omega},-1_{M_\Omega}/d,1_{M_\Omega})\cdot(\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)=\vec{z}_2-\frac{1}{d}\vec{z}_3+\vec{z}_4\\
\nabla_v B(\vec{z}_5)&=1_{M_{\partial \Omega}}&& \Rightarrow \frac{d}{du} B(u)=1_{M_{\partial \Omega}}\cdot \vec{z}_5=\vec{z}_5
\end{align*}
$$
Update features at step $l+1$:
$$
\begin{align}
(\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)_{l+1}&=f(x_{1:M_\Omega})-P((\vec{z}_1,\vec{z}_2,\vec{z}_3,\vec{z}_4)_l)+\frac{d}{du}P(u)\cdot u(x_{1:M_\Omega})_l\\
(\vec{z}_5)_{l+1}&=g(x_{1:M_{\partial\Omega}})-B((\vec{z}_5)_l)+\frac{d}{du}B(u)\cdot u(x_{1:M_{\partial\Omega}})_l
\end{align}
$$
Note that $M=M_\Omega+M_{\partial\Omega}$.

The iteration step becomes
$$
\begin{align}
u(x_{1:M})&=K(x_{1:M},\phi_l)K(\phi_l,\phi_l)^{\dagger}\begin{pmatrix}\vec{z}_1\\\vec{z}_2\\\vec{z}_3\\\vec{z}_4\\\vec{z}_5\\\end{pmatrix}
\end{align}
$$

where $\dagger$ stands for Moore-Penrose inverse of the matrix.

## Parameters

Specifically, we consider the problem for
$$
d=100, \mu=-1/d,\sigma=\sqrt{2}, D=[-0.5,0.5]^{100}, s=0, T=0.5
$$

and

$$
d=250,\mu=-1/d, \sigma=\sqrt{2}, D=[-0.5,0.5]^{250}, s=0, T=0.5
$$

