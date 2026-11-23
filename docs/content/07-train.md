## Learning the path: the PCA example

Training changes $\theta$, which bends the whole path; its end slides towards the minimum of the loss. As an example, take functional PCA. Data $f\sim\mathcal{D}$ (images, fields, signals viewed as functions, mean removed) have covariance $\mathcal{C}=\mathbb{E}[f\otimes f]$. With weights $p_1>p_2>\dots>p_K>0$, minimize

$$\mathcal{L}(\theta)=-\sum_{k=1}^{K}p_k\,\mathbb{E}\,\langle\varphi_k^\theta,f\rangle^2=-\sum_{k=1}^{K}p_k\,\langle\varphi_k^\theta,\mathcal{C}\,\varphi_k^\theta\rangle .$$

### Why the minimum is the PCA basis

Set $p_{K+1}=0$ and sum by parts:

$$\sum_{k=1}^{K}p_k\,\langle\varphi_k,\mathcal{C}\varphi_k\rangle=\sum_{k=1}^{K}(p_k-p_{k+1})\sum_{j\le k}\langle\varphi_j,\mathcal{C}\varphi_j\rangle .$$

Each coefficient $p_k-p_{k+1}$ is positive, and each inner sum is at most the sum of the $k$ largest eigenvalues of $\mathcal{C}$ (Ky Fan), with equality on the top-$k$ eigenspace. So the loss is minimized by the eigenfunctions of $\mathcal{C}$ in order of decreasing eigenvalue, which is functional PCA (the Karhunen–Loève basis). Every prefix of the learned basis is then optimal for its size.

### Training

Expectations are estimated on minibatches and inner products by quadrature. Gradients flow back through every Cayley step, as in a discretized Neural ODE:

$$\theta\leftarrow\theta-\eta\,\nabla_\theta\mathcal{L}(\theta).$$

> In the animation the grey paths are earlier iterates and the loss curve is schematic.
