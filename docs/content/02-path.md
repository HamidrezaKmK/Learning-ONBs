## A path of rotations from Fourier to a learned basis

Rather than parameterizing $Q$ directly, we reach it along a path. Let $Q(t)$, $t\in[0,1]$, solve

$$\frac{d}{dt}Q(t)=K(t)\,Q(t),\qquad Q(0)=I,$$

and move every basis function with it: $e_k^t=Q(t)\,e_k$. The learned basis is the end of the path, $\varphi_k=Q(1)\,e_k$.

### Skew-adjoint generators give isometries

If $K(t)^*=-K(t)$ for all $t$, inner products are constant along the path:

$$\frac{d}{dt}\langle Q f,Qg\rangle=\langle KQf,Qg\rangle+\langle Qf,KQg\rangle=\langle Qf,(K^*+K)\,Qg\rangle=0 .$$

So $\langle Q(t)f,Q(t)g\rangle=\langle f,g\rangle$, and in particular $\langle e_i^t,e_j^t\rangle=\delta_{ij}$ at every $t$. The flow can be run backwards, so $Q(t)$ is invertible and therefore orthogonal. Every point of the path is an orthonormal basis.

### Two neural networks define the generator

Two MLPs $\alpha_\theta(x,t)$ and $\beta_\theta(x,t)$ define a rank-2 generator,

$$K_\theta(t)=\alpha_\theta^t\otimes\beta_\theta^t-\beta_\theta^t\otimes\alpha_\theta^t,\qquad (K_\theta(t)f)(x)=\alpha_\theta^t(x)\,\langle\beta_\theta^t,f\rangle-\beta_\theta^t(x)\,\langle\alpha_\theta^t,f\rangle .$$

It is skew-adjoint by construction: $\langle K f,g\rangle=\langle\beta,f\rangle\langle\alpha,g\rangle-\langle\alpha,f\rangle\langle\beta,g\rangle=-\langle f,Kg\rangle$. One rank-2 generator at each instant is enough. The paper shows that the endpoints $Q(T)$ of such flows are dense in the rotations $e^{K}$ with Hilbert–Schmidt $K$ (see the paper for the precise statement).
