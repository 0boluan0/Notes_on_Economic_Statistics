---
aliases:
  - '等式约束局部最优在约束梯度满秩时必须满足 Lagrange 驻点条件，但该条件本身并不充分'
  - At an equality-constrained local optimum, full-rank constraint gradients imply the Lagrange stationarity condition, which is necessary but not sufficient
student_os: knowledge-atom
atom_id: OPT-MV-003
atom_set: multivariable-optimization
atom_type: necessary-condition
status: source-checked
mastery_state: unassessed
requires:
  - "[[无约束一阶条件]]"
related:
  - "[[等边际原则]]"
  - "[[梯度与水平集]]"
  - "[[乘子不是精确罚金]]"
leads_to:
  - "[[等式约束二阶条件]]"
  - "[[KKT条件]]"
  - "[[影子价格]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# 等式约束局部最优在约束梯度满秩时必须满足 Lagrange 驻点条件，但该条件本身并不充分
<!-- bilingual-en:start -->
*At an equality-constrained local optimum, full-rank constraint gradients imply the Lagrange stationarity condition, which is necessary but not sufficient*
<!-- bilingual-en:end -->

> [!summary] 正则性让目标梯度落入约束法向空间
> 设 $f:\mathbb R^n\to\mathbb R$ 与 $h=(h_1,\ldots,h_m):\mathbb R^n\to\mathbb R^m$ 在 $x^*$ 附近为 $C^1$，$h(x^*)=0$，且 $Dh(x^*)$ 行满秩 $m$。若 $x^*$ 是约束集 $h(x)=0$ 上的局部最大或最小点，则存在 $\nu\in\mathbb R^m$ 使
> $$
> \nabla f(x^*)=Dh(x^*)^T\nu.
> $$
>
> <!-- bilingual-en:start -->
> Let $f:\mathbb R^n\to\mathbb R$ and $h=(h_1,\ldots,h_m):\mathbb R^n\to\mathbb R^m$ be $C^1$ near a feasible point $x^*$, and suppose $Dh(x^*)$ has full row rank $m$. If $x^*$ is a local maximum or minimum subject to $h(x)=0$, then some $\nu\in\mathbb R^m$ satisfies $\nabla f(x^*)=Dh(x^*)^T\nu$.
> <!-- bilingual-en:end -->

## 几何含义
<!-- bilingual-en:start -->
*Geometric meaning*
<!-- bilingual-en:end -->

在正则点，可行集附近像一张光滑曲面。它的切方向 $d$ 满足 $Dh(x^*)d=0$。若沿某个可行切方向仍有 $\nabla f(x^*)^Td\ne0$，选择适当方向就能在一阶上改善目标，与局部最优矛盾。因此目标梯度必须垂直于所有切方向，也就是落在约束梯度张成的法向空间中。

<!-- bilingual-en:start -->
At a regular point, the feasible set is locally a smooth surface. Its tangent directions satisfy $Dh(x^*)d=0$. If $\nabla f(x^*)^Td\ne0$ for a feasible tangent direction, one sign of that direction improves the objective to first order, contradicting local optimality. Hence the objective gradient must be orthogonal to every tangent direction and therefore lie in the normal space spanned by the constraint gradients.
<!-- bilingual-en:end -->

取
$$
\mathcal L(x,\nu)=f(x)-\nu^Th(x),
$$
定理给出的方程可写成 $\nabla_x\mathcal L(x^*,\nu)=0$，再与 $h(x^*)=0$ 联立。乘子符号取决于 Lagrangian 的定义；改用 $f+\nu^Th$ 会把 $\nu$ 整体变号，但最优选择不变。

<!-- bilingual-en:start -->
With $\mathcal L(x,\nu)=f(x)-\nu^Th(x)$, the theorem becomes $\nabla_x\mathcal L(x^*,\nu)=0$ together with $h(x^*)=0$. Multiplier signs depend on the Lagrangian convention: replacing the minus sign by a plus sign reverses $\nu$ without changing the optimizer.
<!-- bilingual-en:end -->

## 为什么需要约束梯度满秩
<!-- bilingual-en:start -->
*Why full-rank constraint gradients matter*
<!-- bilingual-en:end -->

若约束在候选点失去一阶信息，乘子方程可能连必要性都没有。最大化 $f(x)=x$、约束 $h(x)=x^2=0$ 时，唯一可行点 $x^*=0$ 当然是最大点也是最小点；但 $h'(0)=0$，方程 $f'(0)=\nu h'(0)$ 变成 $1=0$，不存在任何乘子。这不是最优点失败，而是约束表示在该点退化。

<!-- bilingual-en:start -->
If the constraint loses its first-order information, the multiplier equation may fail even as a necessary condition. Maximising $f(x)=x$ subject to $h(x)=x^2=0$ leaves the single feasible point $x^*=0$, which is both a maximum and a minimum. Yet $h'(0)=0$, so $f'(0)=\nu h'(0)$ becomes $1=0$ and no multiplier exists. The optimum is not at fault; the constraint representation is degenerate there.
<!-- bilingual-en:end -->

## 为什么驻点条件不充分
<!-- bilingual-en:start -->
*Why stationarity is not sufficient*
<!-- bilingual-en:end -->

Lagrange 方程只排除正则可行切方向上的一阶改善。它不区分最大、最小或鞍点，也不保证候选点是全局解。需要 [[等式约束二阶条件]]、凹性/凸性或直接比较继续验证。经济学中的 [[等边际原则]] 正是这一必要条件的应用，但角点、非光滑偏好或非凹目标仍要另行处理。

<!-- bilingual-en:start -->
The Lagrange equations only rule out first-order improvement along regular feasible tangent directions. They do not distinguish maxima, minima, and saddles or guarantee a global solution. Verification therefore continues with [[等式约束二阶条件|constrained second-order conditions]], concavity/convexity, or direct comparison. The economic [[等边际原则|equal-marginal principle]] is an application of this necessary condition, but corners, nonsmooth preferences, and nonconcave objectives require separate treatment.
<!-- bilingual-en:end -->

这里的 Lagrange 条件只回答等式约束下的一阶必要性。乘子与精确罚函数的区别见 [[乘子不是精确罚金]]；乘子的价值敏感度解释见 [[影子价格]]。

<!-- bilingual-en:start -->
The Lagrange condition here answers only first-order necessity under equality constraints. [[乘子不是精确罚金|The exact-penalty distinction]] and [[影子价格|the shadow-price definition]] explain two separate multiplier interpretations.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么把约束 $x=0$ 改写成 $x^2=0$ 会破坏 Lagrange 定理的使用，尽管可行集没有改变？
>
> <!-- bilingual-en:start -->
> Why can replacing the constraint $x=0$ by $x^2=0$ break the Lagrange theorem even though the feasible set is unchanged?
> <!-- bilingual-en:end -->
>
> **答案：** 新约束在零点梯度为零，失去满秩约束资格；Lagrange 定理依赖约束的一阶表示，而不仅依赖可行点集合。
>
> <!-- bilingual-en:start -->
> **Answer:** The new constraint has zero gradient at the origin and violates the full-rank constraint qualification. The theorem depends on the first-order representation of the constraints, not merely on the feasible set as a collection of points.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*SOFP Lecture Notes*（课程讲义）：直接核对单个与多个等式约束的 Lagrange 定理、梯度满秩条件及课程记号。
- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：交叉核对约束资格下的一阶必要性。

<!-- bilingual-en:start -->
- The EC400 SOFP notes were checked directly for the single- and multiple-equality Lagrange theorem, the full-rank condition, and course notation.
- MIT 6.7220 Lecture 7 was used to cross-check first-order necessity under a constraint qualification.
<!-- bilingual-en:end -->
