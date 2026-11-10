---
aliases:
  - Multiplier uniqueness under LICQ
  - LICQ 下的乘子唯一性
student_os: knowledge-atom
atom_id: OPT-MV-020
atom_set: multivariable-optimization
atom_type: uniqueness-theorem
status: source-checked
requires:
  - "[[LICQ条件]]"
  - "[[KKT条件]]"
related:
  - "[[严格凹不保证乘子唯一]]"
  - "[[绑定不推正乘子]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

<!-- bilingual-en:start -->
# Multiplier uniqueness under LICQ
<!-- bilingual-en:end -->

*LICQ 保证乘子唯一*

<!-- bilingual-en:start -->
At one fixed feasible point, [[KKT条件|KKT multipliers]] describe how constraint gradients combine to balance the objective gradient. The question here is whether **two different coefficient lists** could work for that same point and that same description of the constraints. [[LICQ条件|LICQ]] rules this out through [[线性无关|linear independence]]. It does not select the point or first establish that any such coefficient list exists.
<!-- bilingual-en:end -->

在一个固定可行点，[[KKT条件|KKT 乘子]]描述约束梯度怎样组合起来平衡目标梯度。这里的问题是：对**同一个点、同一套约束表示**，能否有两套不同的系数都成立？[[LICQ条件|LICQ]] 通过[[线性无关]]排除这种情况。它并不替你选出这个点，也不是先证明这样的系数一定存在。

<!-- bilingual-en:start -->
Use real variables $x\in U\subseteq\mathbb R^n$, where $U$ is open and $n\ge1$ is an integer. Consider maximizing $f(x)$ with $h_j(x)=0$ for $j=1,\ldots,p$ and $g_i(x)\le0$ for $i=1,\ldots,m$; all these functions are differentiable at the feasible point $x^*$. Either constraint count may be zero. The [[全微分#实值函数、梯度与全微分|gradients]] below are columns. In the sign convention $\mathcal L=f-\sum_i\lambda_i g_i-\sum_j\nu_j h_j$, a KKT pair $(\lambda,\nu)$ at $x^*$ satisfies
<!-- bilingual-en:end -->

取实变量 $x\in U\subseteq\mathbb R^n$，$U$ 为开集，整数 $n\ge1$。考虑在 $h_j(x)=0$（$j=1,\ldots,p$）、$g_i(x)\le0$（$i=1,\ldots,m$）下最大化 $f(x)$，所有函数均在可行点 $x^*$ 可微。两类约束的条数都可以为零。下式的[[全微分#实值函数、梯度与全微分|梯度]]排成列。采用 $\mathcal L=f-\sum_i\lambda_i g_i-\sum_j\nu_j h_j$ 的号约定，该点的一对 KKT 乘子 $(\lambda,\nu)$ 满足

$$
\nabla f(x^*)=\sum_{i=1}^m\lambda_i\nabla g_i(x^*)+
\sum_{j=1}^p\nu_j\nabla h_j(x^*),\qquad
\lambda_i\ge0,\quad\lambda_i g_i(x^*)=0.
$$

<!-- bilingual-en:start -->
The coefficients $\lambda_i$ belong to inequalities; the equality coefficients $\nu_j$ have no sign restriction. Feasibility is already included in our fixed point assumption. The product equations are [[互补松弛|complementary slackness]]. Let $\mathcal A(x^*)=\{i:g_i(x^*)=0\}$ denote the active inequality indices.
<!-- bilingual-en:end -->

系数 $\lambda_i$ 属于不等式；等式系数 $\nu_j$ 不受符号限制。固定点的假设已经包含可行性；乘积等式就是[[互补松弛]]。记 $\mathcal A(x^*)=\{i:g_i(x^*)=0\}$ 为活动不等式的索引集合。

> [!theorem] Uniqueness at the fixed point · 固定点处的唯一性
> <!-- bilingual-en:start -->
> Suppose KKT multipliers exist at $x^*$. Under LICQ, equality gradients and active inequality gradients are linearly independent. Inactive multipliers are zero by complementary slackness, while the remaining multipliers are the unique coefficients representing the objective gradient in that independent family of normals.
> <!-- bilingual-en:end -->
>
> 设 $x^*$ 已有 KKT 乘子。若等式约束梯度与所有活动不等式约束梯度满足 [[LICQ条件|LICQ]]，则这些乘子唯一。非活动不等式的乘子由互补松弛固定为零；活动约束与等式约束的乘子则是把 $\nabla f(x^*)$ 表示成一组线性无关法向量的系数。

<!-- bilingual-en:start -->
## Why subtracting the equations is enough
<!-- bilingual-en:end -->

## 一行证明

<!-- bilingual-en:start -->
Take any two full KKT pairs at the same $x^*$. If $i\notin\mathcal A(x^*)$, feasibility makes $g_i(x^*)<0$. Complementary slackness gives $\lambda_i g_i(x^*)=\tilde\lambda_i g_i(x^*)=0$; division by this nonzero constraint value gives $\lambda_i=\tilde\lambda_i=0$. Thus only equality and active-inequality coefficients remain in stationarity.
<!-- bilingual-en:end -->

任取同一 $x^*$ 处的两套完整 KKT 乘子。如果 $i\notin\mathcal A(x^*)$，可行性给出 $g_i(x^*)<0$。互补松弛使 $\lambda_i g_i(x^*)=\tilde\lambda_i g_i(x^*)=0$；除以这个非零约束值，得到 $\lambda_i=\tilde\lambda_i=0$。所以驻点方程只剩等式与活动不等式的系数。

<!-- bilingual-en:start -->
Subtracting two stationarity equations gives a zero linear combination of equality and active-inequality gradients. LICQ forces every coefficient difference to vanish. Complementary slackness already fixes inactive multipliers at zero, so the complete multiplier vector is unique.
<!-- bilingual-en:end -->

若 $(\lambda,\nu)$ 与 $(\tilde\lambda,\tilde\nu)$ 都满足同一点的驻点方程，两式相减得到
$$
\sum_{i\in\mathcal A(x^*)}(\lambda_i-\tilde\lambda_i)\nabla g_i(x^*)
+\sum_j(\nu_j-\tilde\nu_j)\nabla h_j(x^*)=0.
$$
LICQ 使这组梯度线性无关，因此每个系数差都为零。再结合非活动乘子全为零，得到完整乘子向量相同。

<!-- bilingual-en:start -->
In this subtraction the coefficient differences need not be nonnegative. LICQ tests **all real coefficients**, so it applies to their differences too. Every gradient retains its constraint index; equal gradients are not merged. Hence any two admissible pairs are equal, which proves there is at most one pair, and the assumed existence makes it exactly one. No convexity or second derivative is used.
<!-- bilingual-en:end -->

相减后的系数差不必非负。LICQ 对**所有实系数**进行检查，因此也适用于这些差。每个梯度保留自己的约束索引，相同梯度不能合并。于是任意两套合格乘子都相同，证明了至多一套；结合已假定的存在性，就恰有一套。证明没有使用凸性或二阶导数。

> [!example] Equality, binding, and slack · 等式、绑定与松弛
> <!-- bilingual-en:start -->
> Maximize $f(x,y)=2x+3y$ subject to $h(x,y)=y=0$, $g_1(x,y)=x\le0$, and $g_2(x,y)=x-1\le0$. At $(0,0)$, only $g_1$ is active, and its gradient $(1,0)^T$ is independent of $\nabla h=(0,1)^T$. Since $g_2=-1$, its multiplier is zero. Stationarity becomes
> <!-- bilingual-en:end -->
>
> 最大化 $f(x,y)=2x+3y$，约束为 $h(x,y)=y=0$、$g_1(x,y)=x\le0$、$g_2(x,y)=x-1\le0$。在 $(0,0)$，只有 $g_1$ 活动；其梯度 $(1,0)^T$ 与 $\nabla h=(0,1)^T$ 无关。因为 $g_2=-1$，其乘子为零。驻点方程变成
>
> $$\binom23=\lambda_1\binom10+\nu\binom01,\qquad
> (\lambda_1,\lambda_2,\nu)=(2,0,3).$$
>
> <!-- bilingual-en:start -->
> These coefficients satisfy all KKT conditions and are the only ones that do so at this point. The point is also a maximizer here, since feasibility gives $f=2x\le0=f(0,0)$, but that last conclusion uses the objective and feasible set directly; it is not supplied by multiplier uniqueness.
> <!-- bilingual-en:end -->
>
> 这组系数满足全部 KKT 条件，而且是该点唯一满足条件的一组。此例中该点确实也是最大点，因为可行性给出 $f=2x\le0=f(0,0)$；但最后这一步直接用了目标与可行集，不是由乘子唯一性推出。

<!-- bilingual-en:start -->
## Limits of the conclusion
<!-- bilingual-en:end -->

## 它没有保证什么

<!-- bilingual-en:start -->
This result assumes multiplier existence. It proves neither optimality nor uniqueness of the primal choice. Conversely, strict concavity may make the choice unique while dependent or duplicated constraints still leave multipliers nonunique.
<!-- bilingual-en:end -->

这条唯一性结论以“乘子已经存在”为前提；乘子存在的条件见 [[KKT必要性]]。乘子唯一也不证明 $x^*$ 最优，更不证明最优选择唯一。反过来，目标严格凹可能使选择唯一，却仍不能修复相关或重复约束造成的乘子不唯一，见 [[严格凹不保证乘子唯一]]。

<!-- bilingual-en:start -->
Dependence permits, but does not force, nonunique KKT multipliers: inequality signs still matter. For example, maximize $-(x-1)^2$ subject to $x\le0$ and $2x\le0$. At the unique maximizer $0$, stationarity is $\lambda_1+2\lambda_2=2$. Both $(2,0)$ and $(0,1)$ are valid; more generally $(2-2t,t)$ works for $0\le t\le1$. With the same constraints but objective $-x^2$, stationarity instead gives $\lambda_1+2\lambda_2=0$, forcing both nonnegative multipliers to zero. LICQ fails in both cases, so it is sufficient, not necessary, for fixed-point multiplier uniqueness.
<!-- bilingual-en:end -->

相关性允许、但不强迫 KKT 乘子不唯一，因为不等式乘子的符号仍受限制。例如最大化 $-(x-1)^2$，约束为 $x\le0$ 与 $2x\le0$。在唯一最大点 $0$，驻点方程为 $\lambda_1+2\lambda_2=2$；$(2,0)$ 与 $(0,1)$ 都有效，更一般地，$0\le t\le1$ 时 $(2-2t,t)$ 都有效。若保持同一约束但把目标换为 $-x^2$，驻点方程变成 $\lambda_1+2\lambda_2=0$，反而迫使两个非负乘子都为零。两例的 LICQ 都失败，因此 LICQ 是同点乘子唯一性的充分条件，并非必要条件。

<!-- bilingual-en:start -->
Uniqueness also uses the **same constraint functions**, not just the same feasible set. Replacing $g_i$ by $2g_i$ rescales its coefficient by $1/2$ while representing the same contribution. If there are no equality or active inequality gradients, LICQ holds for the empty family; any existing KKT multipliers are all zero on inactive constraints, and stationarity requires $\nabla f(x^*)=0$.
<!-- bilingual-en:end -->

唯一性还针对**同一套约束函数**，不只是同一个可行集。把 $g_i$ 改成 $2g_i$，表示同一份贡献的系数会缩为原来的 $1/2$。若没有等式或活动不等式梯度，空族满足 LICQ；任何已存在的 KKT 乘子在非活动约束上均为零，驻点条件要求 $\nabla f(x^*)=0$。

> [!question]- Why one coefficient list? · 为什么只有一套系数？
> <!-- bilingual-en:start -->
> Why does LICQ rule out two different multiplier representations of the same objective gradient?
> <!-- bilingual-en:end -->
>
> LICQ 为什么能排除同一目标梯度的两组不同乘子表示？
>
> <!-- bilingual-en:start -->
> **Answer:** Subtracting the two representations gives a zero linear combination of active normals. LICQ forces every coefficient difference to vanish.
> <!-- bilingual-en:end -->
>
> **答案：** 两组表示相减会给出活动法向量的零线性组合；LICQ 的线性无关迫使所有系数差为零。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- MIT 6.7220 Lecture 7 was checked for LICQ and multiplier uniqueness.
- Stanford CME307/MS&E311 Lecture Note 7 was used to cross-check the normal-vector representation at a regular point.
<!-- bilingual-en:end -->

- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：核对 LICQ、活动梯度与乘子唯一性的关系。
- [Stanford CME307/MS&E311, Lecture Note 7](https://web.stanford.edu/class/msande311/lecture07.pdf)：交叉核对正则点的法向量表示。

<!-- bilingual-en:start -->
- Precise source scope: MIT PDF pp.2–4 gives the KKT equations, inactive-multiplier implication and the indexed LICQ family (including footnote 3 on repetitions); Stanford PDF pp.3, 5, 7–9 gives regularity and the gradient representation. The uniqueness claim is derived above from these equations by subtraction and linear independence, rather than attributed to a separately stated uniqueness theorem in those notes.
- MIT uses minimization with $g\le0$ and Stanford uses minimization with $c\ge0$. This card states its maximization/$g\le0$ sign convention explicitly. All example coefficient equations, signs and complementary products are checked directly; the equality of two fixed-point multiplier vectors is the general proof.
<!-- bilingual-en:end -->

- 精确来源范围：MIT PDF 第2–4页给出 KKT 方程、非活动乘子为零的结论，以及按索引保留的 LICQ 梯度族（含说明重复项的脚注3）；Stanford PDF 第3、5、7–9页给出正则性与梯度表示。本文通过相减和线性无关从这些方程推导唯一性，不把它归作两份讲义中另行陈述的唯一性定理。
- MIT 使用最小化且 $g\le0$，Stanford 使用最小化且 $c\ge0$；本卡显式采用最大化且 $g\le0$ 的号约定。所有例子的系数方程、符号与互补乘积直接核算；两组同点乘子相等的论证才是一般证明。
