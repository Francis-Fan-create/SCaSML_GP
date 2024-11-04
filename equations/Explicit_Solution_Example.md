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
\vec{y}_{domain}=0_{M_{\Omega}}
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

Feature vector $\phi$ have $M=4M_{\partial \Omega}+ M_{\Omega}$ components, which is also the size of the kernel matrix.

Taking derivatives under linearization condition
$$
\begin{align*}
DF(\vec{z}_{k})=\begin{pmatrix}\sigma^2\text{diag}(\vec{z}_5) && 0 && \frac{\sigma^2}{2}I_{M_{\Omega}}&& I_{M_{\Omega}} && \sigma^2\text{diag}(\vec{z}_1)-(\frac1d+\frac{\sigma^2}{2})I_{M_{\Omega}} \\
0 && I_{M_{\partial \Omega}} && 0 && 0 && 0\end{pmatrix}
\end{align*}
$$

Run the algorithm as the following:

Step 1: Construct $\mathbf{y}$
$$
\mathbf{y}=\begin{pmatrix}\vec{y}_{domain}\\g((x,t)_{1:M_{\partial \Omega}}^{\partial \Omega})\end{pmatrix}
$$
Step 2: Solve $\gamma$
$$
\left(DF(\vec{z}^{k})K(\phi,\phi)(DF(\vec{z}^{k}))^{T}\right)\gamma=\mathbf{y}-F(\vec{z}^{k})+DF(\vec{z}^{k})\vec{z}^{k} .
$$
where:
$$
K(\phi,\phi)=\begin{pmatrix}
K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) && \Delta_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& D_{t_y}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&div_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})\\
K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) && \Delta_y K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& D_{t_y}K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&div_y K((x,t_x)_{\partial \Omega}^{1:M_{\partial\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})\\
\Delta_x K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& \Delta_x K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) && \Delta_x\Delta_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&\Delta_x D_{t_y}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&\Delta_x div_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})\\
D_{t_x}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&D_{t_x} K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) &&D_{t_x} \Delta_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& D_{t_x}D_{t_y}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&D_{t_x} div_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})\\
div_x K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&div_x  K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_{\partial \Omega}^{1:M_{\partial\Omega}}) &&div_x  \Delta_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&& div_x  D_{t_y}K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})&&div_x div_y K((x,t_x)_\Omega^{1:M_{\Omega}},(y,t_y)_\Omega^{1:M_{\Omega}})
\end{pmatrix}
$$
Under the RBF kernel we use, for $\tilde{\sigma}=\sigma\cdot\sqrt{d}$, we have:
$$
K(\phi,\phi) = \begin{pmatrix}
e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{t_y-t_x}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{\sum(y-x)}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} \\

e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{t_y-t_x}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{\sum(y-x)}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} \\

(-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (\frac{d(d+2)}{\tilde{\sigma}^4}-\frac{2(d+2)\|x-y\|^2}{\tilde{\sigma}^6}+\frac{\|x-y\|^4}{\tilde{\sigma}^8})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & \frac{t_x-t_y}{\tilde{\sigma}^2}(-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (\frac{(d+1)\sum(y-x)}{\tilde{\sigma}^4}+\frac{\sum(y-x)}{\tilde{\sigma}^6}-\frac{\|y-x\|^2\sum(y-x)}{\tilde{\sigma}^8})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} \\

-\frac{t_x-t_y}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{t_x-t_y}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & \frac{t_y-t_x}{\tilde{\sigma}^2}(-\frac{d}{\tilde{\sigma}^2}+\frac{\|x-y\|^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{(t_x-t_y)^2}{\tilde{\sigma}^4}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & \frac{t_y-t_x}{\tilde{\sigma}^2}(-\frac{\sum(x-y)}{\tilde{\sigma}^2})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} \\

-\frac{\sum(x-y)}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{\sum(x-y)}{\tilde{\sigma}^2}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (\frac{(d+1)\sum(x-y)}{\tilde{\sigma}^4}+\frac{\sum(x-y)}{\tilde{\sigma}^6}-\frac{\|x-y\|^2\sum(x-y)}{\tilde{\sigma}^8})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & -\frac{(t_x-t_y)\sum(x-y)}{\tilde{\sigma}^4}e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}} & (\frac{d}{\tilde{\sigma}^2}-\frac{(\sum(x-y))^2}{\tilde{\sigma}^4})e^{-\frac{\|(x,t_x)-(y,t_y)\|^2}{2\tilde{\sigma}^2}}
\end{pmatrix}
$$
Step 3: Compute $\vec{z}^{k+1}$
$$
\vec{z}^{k+1}=K(\phi,\phi)(DF(\vec{z}^{k}))^{T}\gamma
$$
Step 4: Return the solution
$$
u((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}})=K((x,t)_{\Omega_{infer}}^{1:M_{\Omega_{infer}}},\phi)K(\phi,\phi)^{-1}\vec{z}^{k+1}
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
d=100, \mu=0, \sigma=0.25, D=[-0.5,0.5]^{100}, s=0, T=0.5
$$

and

$$
d=250, \mu=0, \sigma=0.25, D=[-0.5,0.5]^{250}, s=0, T=0.5
$$