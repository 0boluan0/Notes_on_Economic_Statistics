---
aliases:
  - '在二阶连续可微的驻点处，负定 Hessian 推出严格局部极大，但不能仅凭一点的 Hessian 推出全局极大'
  - A negative-definite Hessian at a stationary point gives a strict local maximum, not automatically a global one
student_os: knowledge-atom
atom_id: CALC-MV-010
atom_set: multivariable-differentiation
atom_type: sufficient-condition
status: source-checked
mastery_state: unassessed
requires:
  - "[[多元Taylor近似]]"
related:
  - "[[Hessian 局部极小判据]]"
  - "[[凸优化全局性]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
  - "[[多元优化.canvas|多元优化]]"
---

# 在二阶连续可微的驻点处，负定 Hessian 推出严格局部极大，但不能仅凭一点的 Hessian 推出全局极大
<!-- bilingual-en:start -->
*At a $C^2$ stationary point, a negative-definite Hessian implies a strict local maximum, but a Hessian at one point cannot by itself imply a global maximum*
<!-- bilingual-en:end -->

> [!summary] 负定二次项在每个小非零方向都降低函数值
> 设 $f$ 在 $x_*$ 的邻域为 $C^2$，满足
> $$
> \nabla f(x_*)=0,
> \qquad
> \nabla^2f(x_*)\prec0.
> $$
> 则 $x_*$ 是严格局部极大点：存在 $x_*$ 的某个邻域，使其中每个 $x\ne x_*$ 都满足 $f(x)<f(x_*)$。
>
> <!-- bilingual-en:start -->
> If $f$ is $C^2$ near a stationary point and its Hessian there is negative definite, then the point is a strict local maximiser: every sufficiently close distinct point has a strictly smaller function value.
> <!-- bilingual-en:end -->

## 判据为何成立
<!-- bilingual-en:start -->
*Why the criterion works*
<!-- bilingual-en:end -->

负定性意味着存在 $c>0$，使所有足够小的 $h$ 都有
$$
h^T\nabla^2f(x_*)h\le -c\|h\|^2.
$$
驻点处的 [[多元Taylor近似]] 为
$$
f(x_*+h)-f(x_*)
=\frac12h^T\nabla^2f(x_*)h+o(\|h\|^2).
$$
当 $h$ 足够小时，严格负的二次项支配余项，因此函数值必然下降。

<!-- bilingual-en:start -->
Negative definiteness gives the uniform quadratic bound displayed above. At a stationary point, Taylor expansion leaves the Hessian quadratic term as the leading change. For sufficiently small $h$, that strictly negative term dominates the remainder, forcing the function value below its value at $x_*$.
<!-- bilingual-en:end -->

## 例子与矩阵检查
<!-- bilingual-en:start -->
*Example and matrix check*
<!-- bilingual-en:end -->

对
$$
f(x,y)=-x^2-2y^2,
$$
原点梯度为零，Hessian 为 $\operatorname{diag}(-2,-4)$。任意非零 $h$ 都使二次型为负，所以原点是严格局部极大；由于这个例子在整个空间凹，它同时也是全局极大，但“全局”来自函数的整体结构，而不是只来自原点的一次 Hessian 检查。

<!-- bilingual-en:start -->
For the displayed concave quadratic, the origin is stationary and the Hessian is $\operatorname{diag}(-2,-4)$, so every nonzero quadratic direction is negative. The origin is also globally maximal in this particular example, but that global conclusion comes from the function's structure over the whole domain, not merely from one Hessian evaluation.
<!-- bilingual-en:end -->

仅看对角元为负仍不够。例如
$$
A=\begin{pmatrix}-1&2\\2&-1\end{pmatrix}
$$
的对角元都为负，但特征值为 $1$ 和 $-3$，所以它不定而非负定。必须检查整个对称矩阵的定号。

<!-- bilingual-en:start -->
Negative diagonal entries alone are insufficient. The displayed matrix has negative diagonal entries but eigenvalues $1$ and $-3$, so it is indefinite rather than negative definite. Definiteness must be checked for the whole symmetric matrix.
<!-- bilingual-en:end -->

## 局部不等于全局
<!-- bilingual-en:start -->
*Local is not global*
<!-- bilingual-en:end -->

函数
$$
g(x)=-x^2+x^4
$$
在 $x=0$ 满足 $g'(0)=0$、$g''(0)=-2<0$，所以零点是严格局部极大；但当 $|x|$ 很大时 $g(x)\to\infty$，零点显然不是全局极大。要把局部极大提升为全局极大，需要整个可行域上的凹性或其他全局比较。

<!-- bilingual-en:start -->
The displayed function has a stationary point at zero with negative second derivative, so zero is a strict local maximum. Yet the quartic term sends the function to infinity far from zero, ruling out a global maximum there. A global claim needs concavity over the feasible domain or another global comparison.
<!-- bilingual-en:end -->

> [!question]- 自检：为什么驻点条件不能省？
> 若 Hessian 在 $x_*$ 负定，但 $\nabla f(x_*)\ne0$，为什么仍不能用本判据宣布局部极大？
>
> <!-- bilingual-en:start -->
> Why can the stationarity condition not be omitted? If the Hessian is negative definite at $x_*$ but the gradient is nonzero, why does this criterion not establish a local maximum there?
> <!-- bilingual-en:end -->
>
> **答案：** 非零线性项在足够小的尺度上支配二次项，并给出一个一阶上升方向。负定 Hessian 只描述局部弯曲，不能抵消未消失的斜率。
>
> <!-- bilingual-en:start -->
> **Answer:** A nonzero linear term dominates the quadratic term at sufficiently small scales and provides a first-order ascent direction. Negative curvature does not cancel a slope that has not vanished.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*SOFP Slides Lecture 2*（课程讲义） slides 10–14：直接核对 $C^2$ 驻点、负定 Hessian 与严格局部极大的课程判据。
- [[多元Taylor近似]]：核验二次项支配余项的推导；这里据此使用“负定推出严格局部极大”的判据，并保留局部而非全局的边界。

<!-- bilingual-en:start -->
- EC400 SOFP Slides Lecture 2, slides 10–14, was checked directly for the $C^2$ stationary-point condition and the negative-definite Hessian criterion for a strict local maximum.
- [[多元Taylor近似|The multivariable Taylor expansion]] supports the argument that the quadratic term dominates the remainder; the criterion here then yields a strict local maximum while remaining local rather than global.
<!-- bilingual-en:end -->
