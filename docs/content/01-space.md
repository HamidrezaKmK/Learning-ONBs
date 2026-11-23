## The space of bases

Let $H=L^2(\Omega)$ with $\langle f,g\rangle=\int_\Omega f(x)\,g(x)\,dx$. An ordered orthonormal basis is a sequence $\varphi_1,\varphi_2,\dots$ with

$$\langle \varphi_i,\varphi_j\rangle=\delta_{ij},\qquad f=\sum_{k\ge 1}\langle \varphi_k,f\rangle\,\varphi_k\quad\text{for every } f\in H.$$

Fix a reference basis $e_1,e_2,\dots$, for example the Fourier basis. Every other orthonormal basis is its image under an orthogonal operator, $\varphi_k=Q\,e_k$ with $Q^*Q=QQ^*=I$, and every such $Q$ gives one. So searching over bases is searching over the orthogonal group of $H$. The sphere in the animation is a cartoon of that group, labelled $\mathrm{SO}(\infty)$; each point is one whole basis.

### What makes a basis good

Bases are ordered, and in practice we keep only the first few coefficients. A loss $\mathcal{L}$ scores a basis by how well it serves a task. For data $f\sim\mathcal{D}$, keeping the first $K$ terms leaves the error

$$\mathbb{E}\,\Big\|f-\sum_{k\le K}\langle\varphi_k,f\rangle\varphi_k\Big\|^2=\mathbb{E}\|f\|^2-\sum_{k\le K}\mathbb{E}\,\langle\varphi_k,f\rangle^2 ,$$

so a good basis puts as much energy as possible in its first functions. The colours and level sets on the sphere show such a loss: red is low, silver is high.

### Why Fourier is not the answer

The Fourier basis diagonalizes translation-invariant operators: stationary covariances, the Laplacian on a periodic box. Most data and most domains are not like that. For faces, digits, flows or PDEs on a shape, energy spreads over many Fourier coefficients, so truncation is wasteful. Fourier is a basis, often a decent one, but not the optimum. A random basis is usually worse still.

> The loss landscape on the sphere is schematic: it illustrates the idea and is not computed from data. The functions drawn on the cards are real rotations of sines.
