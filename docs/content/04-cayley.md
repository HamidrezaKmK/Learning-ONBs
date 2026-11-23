## Every function turns, and how a step is computed

The animation stops with four Fourier functions, each turned in the same plane. Each moves by an amount set by its own projection: a function almost orthogonal to the plane barely changes, one inside it turns by the full angle.

### Inner products by quadrature

Every $e_k^t$ is a function we can evaluate anywhere, because it is $e_k$ plus finitely many multiples of earlier $\alpha$'s and $\beta$'s. The two inner products a step needs are computed at quadrature points $x_1,\dots,x_N$ with weights $w_i$:

$$\langle\alpha,f\rangle\approx\sum_{i=1}^N w_i\,\alpha(x_i)\,f(x_i).$$

### The Cayley step

We do not exponentiate. Each step uses the Cayley transform, with $K_\ell$ the generator at the step's midpoint:

$$\exp(\Delta t\,K_\ell)\approx\mathrm{Cay}(\Delta t\,K_\ell)=\big(I-\tfrac{\Delta t}{2}K_\ell\big)^{-1}\big(I+\tfrac{\Delta t}{2}K_\ell\big).$$

It is exactly orthogonal for any step size. With $h=\Delta t/2$, the factors $I\pm hK$ commute and $(I+hK)^*=I-hK$, so

$$\mathrm{Cay}^*\,\mathrm{Cay}=(I-hK)(I+hK)^{-1}(I-hK)^{-1}(I+hK)=I .$$

### Only a 2×2 solve

Write $K=UJU^*$ with $U c=c_1\alpha+c_2\beta$, $U^*f=(\langle\alpha,f\rangle,\langle\beta,f\rangle)$, $J=\begin{pmatrix}0&1\\-1&0\end{pmatrix}$ and $G=U^*U$ the $2\times2$ Gram matrix of $\alpha,\beta$. The push-through identity turns the operator inverse into a $2\times 2$ one:

$$\mathrm{Cay}(\Delta t\,K)=I+2h\,U\,(I_2-hJG)^{-1}J\,U^* .$$

Applied to one function, a step is two inner products, a $2\times2$ solve and a sum of two functions: $O(N)$ work. When $\alpha,\beta$ are orthonormal, the Cayley step turns the plane by $2\arctan(\Delta t/2)=\Delta t-\Delta t^3/12+\dots$ instead of $\Delta t$; it is orthogonal either way. The code generalizes this to generators of higher finite rank, with a small dense solve.
