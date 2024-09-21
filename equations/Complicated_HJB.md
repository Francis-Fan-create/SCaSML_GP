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
L_3(u):u&\rightarrow \Delta_x u\\
L_4(u):u&\rightarrow u
\end{align}
$$
We then derive
$$
\begin{align}
P(v_1,v_2,v_3)&=v_1-\frac{1}{d}v_2+v_3\\
B(v)&=v
\end{align}
$$
Thus $Q_\Omega=3, Q=4$. 

Feature vector $\phi$ have $N=M_\Omega+M_\Omega+M_\Omega+(M-M_\Omega)$ components, which is also the size of the kernel matrix.

Taking derivatives
$$
\begin{align*}
P'(v_1,v_2,v_3)&=(1,-1/d,1)&&\Rightarrow P'(u)=(1,-1/d,1)\cdot(z_1,z_2,z_3)=(z_1,-\frac1d z_2,z_3)\\
B'(v)&=1&&\Rightarrow B'(u)=1\cdot z_4=z_4
\end{align*}
$$
The iteration step becomes
$$
\begin{align}
u^{\ell+1}(x)&=K(x,\boldsymbol{\phi}^{l})[K(\boldsymbol{\phi}^{l},\boldsymbol{\phi}^{l})+\eta\mathrm{diag}(K(\boldsymbol{\phi}^{l},\boldsymbol{\phi}^{l}))]^{-1}\begin{pmatrix}\left(f-u^{\ell}+\mathcal{P}^{\prime}(u^{\ell})u^{\ell}\right)|_{\mathbf{s}_{\Omega}}\\\left(g-u^{\ell}+\mathcal{B}^{\prime}(u^{\ell})u^{\ell}\right)|_{\mathbf{s}_{\partial\Omega}}\end{pmatrix}
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

