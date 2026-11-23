## One small step: what $\exp(\Delta t\,K)$ does

Over a short interval the generator is nearly constant, so

$$Q(t+\Delta t)\approx\exp\!\big(\Delta t\,K_\theta(t)\big)\,Q(t).$$

Write $\alpha=\alpha_\theta^t$, $\beta=\beta_\theta^t$, and take them orthonormal first. Then

$$K\alpha=-\beta,\qquad K\beta=\alpha,\qquad Kf=0\ \text{ for } f\perp\mathrm{span}\{\alpha,\beta\}.$$

$K$ is the generator of rotations in the plane $\mathrm{span}\{\alpha,\beta\}$, and its exponential rotates that plane by the angle $\Delta t$ while fixing everything orthogonal to it. With $a=\langle\alpha,f\rangle$ and $b=\langle\beta,f\rangle$,

$$\exp(\Delta t\,K)\,f=f+(\cos\Delta t-1)\,(a\,\alpha+b\,\beta)+\sin\Delta t\,(b\,\alpha-a\,\beta).$$

This is an operator on an infinite-dimensional space, but it sees each function only through two numbers. The right panel of the animation is the plane $\mathrm{span}\{\alpha,\beta\}$, and each arrow is a function's projection $(a,b)$.

For general $\alpha,\beta$ the picture is the same with a different speed: restricted to the plane, $K$ has eigenvalues $\pm i\omega$ with

$$\omega^2=\|\alpha\|^2\|\beta\|^2-\langle\alpha,\beta\rangle^2,$$

so the plane turns by the angle $\omega\,\Delta t$.
