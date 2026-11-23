## The whole frame turns at once

The same operator $C=\mathrm{Cay}(\Delta t\,K_\ell)$ moves every function of the basis,

$$e_k^{t+\Delta t}=C\,e_k^{t}\qquad\text{for all }k,$$

including the infinitely many that are never drawn. Because $C$ is orthogonal, the Gram matrix of the frame never changes:

$$\langle e_i^{t+\Delta t},e_j^{t+\Delta t}\rangle=\langle Ce_i^{t},Ce_j^{t}\rangle=\langle e_i^{t},C^*C\,e_j^{t}\rangle=\langle e_i^{t},e_j^{t}\rangle=\delta_{ij}.$$

No Gram–Schmidt, no penalty term and no projection back onto a constraint is needed, whatever the step size. The grid in the animation is computed live from the moving functions and stays the identity.

When the inner products are computed by quadrature, the same argument holds for the discrete inner product $\langle f,g\rangle_N=\sum_i w_i f(x_i)g(x_i)$: build $C$ with it and the frame is orthonormal in $\langle\cdot,\cdot\rangle_N$ up to round-off.
