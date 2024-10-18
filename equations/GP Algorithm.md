# GP Algorithm

Step 1: Construct $\mathbf{y}$
$$
\mathbf{y}=\begin{pmatrix}\vec{y}_{domain}\\g(x_{1:M_{\partial \Omega}}^{\partial \Omega})\end{pmatrix}
$$
Step 2: Solve $\gamma$
$$
\left(DF(\vec{z}^{k})K(\phi,\phi)(DF(\vec{z}^{k}))^{T}\right)\gamma=\mathbf{y}-F(\vec{z}^{k})+DF(\vec{z}^{k})\vec{z}^{k} .
$$
Step 3: Compute $\vec{z}^{k+1}$
$$
\vec{z}^{k+1}=K(\phi,\phi)(DF(\vec{z}^{k}))^{T}\gamma
$$
Step 5: Return the solution
$$
u(x)=\vec{z}_{1:M_{\Omega}}
$$
