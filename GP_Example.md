# GP_Example

We solve the following PDE

$$
\Delta_x u(x,t)+d\sin(\sum_{i=1}^d x_i)=0,t\in[s,T],x\in D⊂\mathbb{R}^d
$$

whose terminal condition is given by


$$
u(x,T)=g(x):=\sin\sum_{i=1}^d x_i,
$$

without any boundary constraints.

The nonlinear term is given by
$$
F(u,z)(x,t)=d\sin(\sum_{i=1}^d x_i)
$$


This PDE has an explicit solution at time $t$
$$
u(x,t)=\sin\sum_{i=1}^d x_i.
$$

which is our target in this section.



Specifically, we consider the problem for

$$
d=100, \mu=0,\sigma=\sqrt{2}, D=[-0.5,0.5]^{100}, s=0, T=0.5
$$

and

$$
d=250,\mu=0, \sigma=\sqrt{2}, D=[-0.5,0.5]^{250}, s=0, T=0.5
$$