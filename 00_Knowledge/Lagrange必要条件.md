---
aliases:
  - Lagrange multiplier condition
  - Equality-constrained first-order necessary condition
  - 等式约束一阶必要条件
student_os: knowledge-atom
atom_id: OPT-MV-003
atom_set: multivariable-optimization
atom_type: necessary-condition
status: source-checked
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

<!-- bilingual-en:start -->
# Lagrange multiplier condition
<!-- bilingual-en:end -->

*Lagrange 乘子必要条件*

<!-- bilingual-en:start -->
When a choice must obey exact equations, an objective can reach its best feasible value even though its gradient is not zero. The **Lagrange multiplier condition** says that, at a suitably regular local optimum, the objective's gradient can be assembled from the constraint gradients. It is a way to find candidates while respecting the directions in which movement is allowed.
<!-- bilingual-en:end -->

当选择必须满足若干精确等式时，即使目标的梯度不为零，也可能已经取得最好的可行值。**Lagrange 乘子条件**说：在具有适当正则性的局部最优点，目标梯度可以由约束梯度组合出来。它帮助我们在允许移动的方向内寻找候选点。

<!-- bilingual-en:start -->
## A budget line before the general notation
<!-- bilingual-en:end -->

*先用一条预算线看问题*

<!-- bilingual-en:start -->
Suppose we maximise $f(x,y)=xy$ subject to $x+4y=16$, with real $x,y$. The equation leaves one free choice: $y=(16-x)/4$. Substitution gives
<!-- bilingual-en:end -->

考虑在 $x+4y=16$ 下最大化 $f(x,y)=xy$，其中 $x,y$ 为实数。等式只留下一个自由选择：$y=(16-x)/4$。代入得到

$$f\left(x,\frac{16-x}{4}\right)=4x-\frac{x^2}{4}
=16-\frac{(x-8)^2}{4}.$$

<!-- bilingual-en:start -->
The unique maximum is $(8,2)$, with value $16$. Yet the [[全微分#实值函数、梯度与全微分|gradient]], the column of first partial derivatives, is $\nabla f(8,2)=(2,8)^T$, not zero. Write the constraint as $h=x+4y-16=0$. Its gradient is $(1,4)^T$, and $\nabla f(8,2)=2\nabla h(8,2)$. A feasible velocity $d=(d_x,d_y)^T$ obeys $d_x+4d_y=0$, so the first-order objective change $2d_x+8d_y$ is zero. The uphill directions of the unconstrained objective are unavailable along this line.
<!-- bilingual-en:end -->

唯一最大点为 $(8,2)$，最大值为 $16$。但把一阶偏导排成列向量的[[全微分#实值函数、梯度与全微分|梯度]]为 $\nabla f(8,2)=(2,8)^T$，并不为零。把约束写成 $h=x+4y-16=0$，其梯度为 $(1,4)^T$，于是 $\nabla f(8,2)=2\nabla h(8,2)$。可行速度 $d=(d_x,d_y)^T$ 满足 $d_x+4d_y=0$，所以目标的一阶变化 $2d_x+8d_y$ 为零。无约束目标的上升方向在这条直线上不可用。

<!-- bilingual-en:start -->
## What the condition assumes
<!-- bilingual-en:end -->

*先分清对象与假设*

<!-- bilingual-en:start -->
Now $x\in\mathbb R^n$ is a vector of choices, $f(x)$ is the scalar objective, and $h_j(x)=0$ for $j=1,\ldots,m$ are the exact restrictions. A point satisfying all of them is **feasible**. A [[局部极值与绝对极值|local optimum]] compares only nearby feasible points. Work on an open neighbourhood of $x^*$; $C^1$ means continuous first partial derivatives there. The [[Jacobian矩阵|Jacobian]] $A=Dh(x^*)$ is an $m\times n$ matrix whose $j$th row is $\nabla h_j(x^*)^T$. Full row rank means these $m$ rows are [[线性无关|linearly independent]], so necessarily $m\le n$. This is a condition on the constraint representation, not a statement that a candidate is optimal.
<!-- bilingual-en:end -->

现在 $x\in\mathbb R^n$ 是选择向量，$f(x)$ 是标量目标，$h_j(x)=0$（$j=1,\ldots,m$）是必须精确满足的限制。满足全部限制的点称为**可行点**。[[局部极值与绝对极值|局部最优]]只比较附近的可行点。在 $x^*$ 的开邻域内讨论；$C^1$ 表示这里的一阶偏导连续。[[Jacobian矩阵]] $A=Dh(x^*)$ 是 $m\times n$ 矩阵，第 $j$ 行为 $\nabla h_j(x^*)^T$。行满秩表示这 $m$ 行[[线性无关]]，因此必有 $m\le n$。这是约束表示的条件，不是候选点已经最优的声明。

> [!important] The necessary condition · 一阶必要条件
> <!-- bilingual-en:start -->
> Let $f:\mathbb R^n\to\mathbb R$ and $h=(h_1,\ldots,h_m):\mathbb R^n\to\mathbb R^m$ be $C^1$ near a feasible point $x^*$, and suppose $Dh(x^*)$ has full row rank $m$. If $x^*$ is a local maximum or minimum subject to $h(x)=0$, then some $\nu\in\mathbb R^m$ satisfies $\nabla f(x^*)=Dh(x^*)^T\nu$.
> <!-- bilingual-en:end -->
>
> 设 $f:\mathbb R^n\to\mathbb R$ 与 $h=(h_1,\ldots,h_m):\mathbb R^n\to\mathbb R^m$ 在 $x^*$ 附近为 $C^1$，$h(x^*)=0$，且 $Dh(x^*)$ 行满秩 $m$。若 $x^*$ 是约束集 $h(x)=0$ 上的局部最大或最小点，则存在 $\nu\in\mathbb R^m$ 使
> $$
> \nabla f(x^*)=Dh(x^*)^T\nu.
> $$

<!-- bilingual-en:start -->
## Geometric meaning
<!-- bilingual-en:end -->

## 几何含义

<!-- bilingual-en:start -->
At a regular point, the feasible set is locally a smooth surface. Its tangent directions satisfy $Dh(x^*)d=0$. If $\nabla f(x^*)^Td\ne0$ for a feasible tangent direction, one sign of that direction improves the objective to first order, contradicting local optimality. Hence the objective gradient must be orthogonal to every tangent direction and therefore lie in the normal space spanned by the constraint gradients.
<!-- bilingual-en:end -->

在正则点，可行集附近像一张光滑曲面。它的切方向 $d$ 满足 $Dh(x^*)d=0$。若沿某个可行切方向仍有 $\nabla f(x^*)^Td\ne0$，选择适当方向就能在一阶上改善目标，与局部最优矛盾。因此目标梯度必须垂直于所有切方向，也就是落在约束梯度张成的法向空间中。

<!-- bilingual-en:start -->
With $\mathcal L(x,\nu)=f(x)-\nu^Th(x)$, the theorem becomes $\nabla_x\mathcal L(x^*,\nu)=0$ together with $h(x^*)=0$. Multiplier signs depend on the Lagrangian convention: replacing the minus sign by a plus sign reverses $\nu$ without changing the optimizer.
<!-- bilingual-en:end -->

取
$$
\mathcal L(x,\nu)=f(x)-\nu^Th(x),
$$
定理给出的方程可写成 $\nabla_x\mathcal L(x^*,\nu)=0$，再与 $h(x^*)=0$ 联立。乘子符号取决于 Lagrangian 的定义；改用 $f+\nu^Th$ 会把 $\nu$ 整体变号，但最优选择不变。

<!-- bilingual-en:start -->
The [[梯度与水平集|level-set picture]] explains one constraint. For several, a tangent vector is the velocity $d=\gamma'(0)$ of a differentiable feasible curve $\gamma$ with $\gamma(0)=x^*$. Differentiating $h(\gamma(t))=0$ gives $Ad=0$. The converse uses the rank assumption: the [[隐函数定理|implicit function theorem]] lets us solve for $m$ coordinates in terms of the other $n-m$. Every vector in the [[零空间|nullspace]] $\ker A$ is then the velocity of a curve in this local graph. Thus the tangent space really is $T=\ker A$; a straight segment $x^*+td$ itself need not stay feasible.
<!-- bilingual-en:end -->

[[梯度与水平集]]说明了单个约束的图像。多个约束下，切向量是可微可行曲线 $\gamma$ 在 $\gamma(0)=x^*$ 处的速度 $d=\gamma'(0)$。对 $h(\gamma(t))=0$ 求导得到 $Ad=0$。反方向需要满秩假设：[[隐函数定理]]允许把 $m$ 个坐标局部写成其余 $n-m$ 个坐标的函数。[[零空间]] $\ker A$ 中的每个向量都可以作为这张局部图像上一条曲线的速度。因此切空间确实是 $T=\ker A$；直线段 $x^*+td$ 本身却未必保持可行。

<!-- bilingual-en:start -->
Along any such two-sided curve, the one-variable function $f(\gamma(t))$ has a local extremum at $0$, so its derivative is zero. Consequently $\nabla f(x^*)$ is perpendicular to $\ker A$. The [[基本子空间正交补|orthogonal-complement identity]] $(\ker A)^\perp=\operatorname{range}(A^T)$ now supplies $\nu$ with $\nabla f=A^T\nu$. Full row rank also makes this multiplier unique at the specified point, since $A^T(\nu_1-\nu_2)=0$ forces $\nu_1=\nu_2$. This does not make the stationary point itself unique.
<!-- bilingual-en:end -->

沿这样的双侧曲线，单变量函数 $f(\gamma(t))$ 在 $0$ 处有局部极值，因此其导数为零。于是 $\nabla f(x^*)$ 垂直于 $\ker A$。由[[基本子空间正交补]]恒等式 $(\ker A)^\perp=\operatorname{range}(A^T)$，存在 $\nu$ 使 $\nabla f=A^T\nu$。行满秩还保证指定点上的乘子唯一，因为 $A^T(\nu_1-\nu_2)=0$ 迫使 $\nu_1=\nu_2$。这并不保证驻点本身唯一。

<!-- bilingual-en:start -->
The entries of $\nu$ are unrestricted real numbers because these are equality constraints. Solving $\nabla_x\mathcal L=0$ and $h=0$ produces candidates; it does not ask us to maximise $\mathcal L$ jointly over $x$ and $\nu$. With no constraints ($m=0$), the equation reduces to the [[无约束一阶条件|unconstrained first-order condition]] $\nabla f=0$. If $m=n$ and $A$ is invertible, the feasible point is locally isolated, so every objective has a local maximum and minimum relative to that nearby feasible set.
<!-- bilingual-en:end -->

这些都是等式约束，因此 $\nu$ 的分量可以是任意实数。联立 $\nabla_x\mathcal L=0$ 与 $h=0$ 是寻找候选点，并不是对 $x,\nu$ 联合最大化 $\mathcal L$。没有约束（$m=0$）时，方程退化为[[无约束一阶条件]] $\nabla f=0$。若 $m=n$ 且 $A$ 可逆，可行点在局部是孤立的，所以相对于附近可行集，任何目标都在此取得局部最大与最小。

<!-- bilingual-en:start -->
## Why full-rank constraint gradients matter
<!-- bilingual-en:end -->

## 为什么需要约束梯度满秩

<!-- bilingual-en:start -->
If the constraint loses its first-order information, the multiplier equation may fail even as a necessary condition. Maximising $f(x)=x$ subject to $h(x)=x^2=0$ leaves the single feasible point $x^*=0$, which is both a maximum and a minimum. Yet $h'(0)=0$, so $f'(0)=\nu h'(0)$ becomes $1=0$ and no multiplier exists. The optimum is not at fault; the constraint representation is degenerate there.
<!-- bilingual-en:end -->

若约束在候选点失去一阶信息，乘子方程可能连必要性都没有。最大化 $f(x)=x$、约束 $h(x)=x^2=0$ 时，唯一可行点 $x^*=0$ 当然是最大点也是最小点；但 $h'(0)=0$，方程 $f'(0)=\nu h'(0)$ 变成 $1=0$，不存在任何乘子。这不是最优点失败，而是约束表示在该点退化。

<!-- bilingual-en:start -->
Failure of full rank removes this guarantee; it does not prove that multipliers cannot exist. For the same constraint $x^2=0$ but objective $f(x)=-x^2$, the equation at zero is $0=\nu\,0$, true for every $\nu$. The first example has no multiplier, the second has infinitely many: neither violates the theorem because neither satisfies its rank assumption.
<!-- bilingual-en:end -->

失去满秩只会撤去这项保证，并不能证明乘子一定不存在。保持约束 $x^2=0$，把目标改成 $f(x)=-x^2$，零点处的方程就是 $0=\nu\,0$，每个 $\nu$ 都满足。前例没有乘子，后例有无穷多个；两者都不满足定理的满秩假设，因此都没有违背定理。

<!-- bilingual-en:start -->
## Why stationarity is not sufficient
<!-- bilingual-en:end -->

## 为什么驻点条件不充分

<!-- bilingual-en:start -->
The Lagrange equations only rule out first-order improvement along regular feasible tangent directions. They do not distinguish maxima, minima, and saddles or guarantee a global solution. Verification therefore continues with [[等式约束二阶条件|constrained second-order conditions]], concavity/convexity, or direct comparison. The economic [[等边际原则|equal-marginal principle]] is an application of this necessary condition, but corners, nonsmooth preferences, and nonconcave objectives require separate treatment.
<!-- bilingual-en:end -->

Lagrange 方程只排除正则可行切方向上的一阶改善。它不区分最大、最小或鞍点，也不保证候选点是全局解。需要 [[等式约束二阶条件]]、凹性/凸性或直接比较继续验证。经济学中的 [[等边际原则]] 正是这一必要条件的应用，但角点、非光滑偏好或非凹目标仍要另行处理。

<!-- bilingual-en:start -->
The Lagrange condition here answers only first-order necessity under equality constraints. [[乘子不是精确罚金|The exact-penalty distinction]] and [[影子价格|the shadow-price definition]] explain two separate multiplier interpretations.
<!-- bilingual-en:end -->

这里的 Lagrange 条件只回答等式约束下的一阶必要性。乘子与精确罚函数的区别见 [[乘子不是精确罚金]]；乘子的价值敏感度解释见 [[影子价格]]。

<!-- bilingual-en:start -->
For example, maximise $f(x,y)=x^2$ subject to $y=0$. The origin satisfies the multiplier equations with $\nu=0$, yet any nearby nonzero $x$ gives a larger feasible value. It is a constrained minimum, not a maximum. Candidate equations and verification of the requested optimum are separate steps.
<!-- bilingual-en:end -->

例如在 $y=0$ 下最大化 $f(x,y)=x^2$。原点以 $\nu=0$ 满足乘子方程，但附近任意非零 $x$ 都给出更大的可行值。它是约束最小点，而非最大点。列出候选方程和确认所求类型的最优点是两个步骤。

> [!question] Check the representation · 检查约束表示
> <!-- bilingual-en:start -->
> Why can replacing the constraint $x=0$ by $x^2=0$ break the Lagrange theorem even though the feasible set is unchanged?
> <!-- bilingual-en:end -->
>
> 为什么把约束 $x=0$ 改写成 $x^2=0$ 会破坏 Lagrange 定理的使用，尽管可行集没有改变？
>
> <!-- bilingual-en:start -->
> **Answer:** The new constraint has zero gradient at the origin and violates the full-rank constraint qualification. The theorem depends on the first-order representation of the constraints, not merely on the feasible set as a collection of points.
> <!-- bilingual-en:end -->
>
> **答案：** 新约束在零点梯度为零，失去满秩约束资格；Lagrange 定理依赖约束的一阶表示，而不仅依赖可行点集合。

## 来源与核验

<!-- bilingual-en:start -->
### Sources and verification

- The EC400 SOFP notes were checked directly for the single- and multiple-equality Lagrange theorem, the full-rank condition, and course notation.
- MIT 6.7220 Lecture 7 was used to cross-check first-order necessity under a constraint qualification.

- [Chachuat, *Nonlinear and Dynamic Optimization*, Chapter 1, PDF pp.23–25, Definitions 1.46–1.47, Lemma 1.48 and Theorem 1.50](https://www.epfl.ch/labs/la/wp-content/uploads/2018/08/ic-32_lectures-1-8.pdf#page=23): tangent velocities, the kernel description at a regular point, and existence and uniqueness of equality multipliers. Here the smoothness assumption is explicitly $C^1$ on a neighbourhood; equation counting alone is not used to claim a unique stationary point.
- [MIT 18.022, *The implicit function theorem*, PDF pp.1–2](https://math.mit.edu/~larsh/teaching/F2007/handouts/implicit.pdf): the local graph used in the tangent-space argument. The budget and degenerate-constraint examples are checked directly above.
- Precise course locations: SOFP PDF/print pp.25–27; MIT 6.7220 Lecture 7 pp.2–4, especially Theorem L7.3. MIT uses the opposite multiplier sign for minimisation; the equality statement agrees after changing signs.
<!-- bilingual-en:end -->

- LSE EC400，*SOFP Lecture Notes*（课程讲义）：直接核对单个与多个等式约束的 Lagrange 定理、梯度满秩条件及课程记号。
- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：交叉核对约束资格下的一阶必要性。

- Chachuat《Nonlinear and Dynamic Optimization》第1章 PDF23–25页，定义1.46–1.47、引理1.48与定理1.50：切向量作为速度、正则点的核空间描述，以及等式乘子的存在与唯一。本文明确使用邻域内 $C^1$ 假设；不以方程与未知量个数相同来断言驻点唯一。
- MIT 18.022《The implicit function theorem》PDF1–2页：切空间论证中的局部图像。预算线与退化约束例均在正文直接计算。
- 课程精确位置：SOFP PDF／印本25–27页；MIT 6.7220 Lecture 7 第2–4页，尤其定理L7.3。MIT 的最小化记号使用相反乘子号，等式版本换号后相同。
