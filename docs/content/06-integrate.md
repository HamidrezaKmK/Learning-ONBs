## Integrating the path

With $L$ steps of size $\Delta t=1/L$, the learned rotation is a product of Cayley factors,

$$Q_\theta=C_L\cdots C_2\,C_1,\qquad C_\ell=\mathrm{Cay}\big(\Delta t\,K_\theta(t_\ell+\tfrac{\Delta t}{2})\big),$$

and the learned basis is $\varphi_k=Q_\theta\,e_k$. Three properties follow:

- **Exactly orthonormal for every $\theta$.** Each factor is orthogonal, so the optimizer moves freely over bases and never has to enforce a constraint.
- **Defined at every point.** Each $\varphi_k$ is $e_k$ plus a finite combination of the functions $\alpha_\theta^{t_\ell},\beta_\theta^{t_\ell}$, so it can be evaluated at any $x$ and any resolution. No grid is fixed.
- **Cheap.** Moving $K$ functions costs $O(L\,K\,N)$ for $N$ quadrature points.

In the animation the frame of Fourier cards rides the path from the start to the learned basis.
