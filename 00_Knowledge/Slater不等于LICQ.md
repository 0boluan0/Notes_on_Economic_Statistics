---
aliases:
  - 'Slater 检查凸问题是否存在严格可行点，LICQ 检查候选点的活动约束梯度，两者不是等价约束资格'
  - Slater checks for a strictly feasible point in a convex problem whereas LICQ checks active-constraint gradients at a candidate, so the qualifications are not equivalent
student_os: knowledge-atom
atom_id: OPT-MV-023
atom_set: multivariable-optimization
atom_type: concept-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[Slater条件]]"
  - "[[LICQ条件]]"
related:
  - "[[KKT必要性]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# Slater 检查凸问题是否存在严格可行点，LICQ 检查候选点的活动约束梯度，两者不是等价约束资格
<!-- bilingual-en:start -->
*Slater checks for a strictly feasible point in a convex problem whereas LICQ checks active-constraint gradients at a candidate, so the qualifications are not equivalent*
<!-- bilingual-en:end -->

> [!summary] 两个条件回答不同问题
> [[Slater条件]] 在整个凸可行性问题中寻找一个严格可行点，该点不必最优；它是全局的可行性条件。[[LICQ条件]] 在指定可行点检查等式梯度与活动不等式梯度是否线性无关；它是局部且依赖约束表示的一阶几何条件。
>
> <!-- bilingual-en:start -->
> Slater searches a convex feasibility problem for a strictly feasible point, which need not be optimal. LICQ tests linear independence of equality and active-inequality gradients at a specified feasible point. One is global strict feasibility; the other is local first-order geometry and depends on the constraint representation.
> <!-- bilingual-en:end -->

## 一个条件成立而另一个失败
<!-- bilingual-en:start -->
*One can hold while the other fails*
<!-- bilingual-en:end -->

约束
$$
x\le1,
\qquad
2x\le2
$$
在 $x=0$ 有严格可行点，所以 Slater 成立。在边界点 $x=1$，两条约束都活动，而梯度 $1$ 与 $2$ 线性相关，所以 LICQ 失败。重复第二条约束没有改变可行集或 Slater，却破坏了 LICQ；这已经否定两者等价。

<!-- bilingual-en:start -->
The duplicate constraints $x\le1$ and $2x\le2$ have a strictly feasible point, so Slater holds. At the boundary point one, both active gradients are dependent, so LICQ fails. Duplicating the constraint leaves the feasible set and Slater unchanged but destroys LICQ.
<!-- bilingual-en:end -->

LICQ 还可以用于非凸局部问题，而标准 Slater 强对偶定理并不适用。例如非线性等式 $x^2+y^2=1$ 在 $(1,0)$ 的梯度非零，局部等式资格成立；但这不是含仿射等式的凸规划，不能把它称为 Slater 的另一种写法。

<!-- bilingual-en:start -->
LICQ also applies to local nonconvex problems where the standard Slater strong-duality theorem is inapplicable. The nonlinear equality $x^2+y^2=1$ is regular at $(1,0)$, yet it is not an affine-equality convex program and LICQ there is not a version of Slater.
<!-- bilingual-en:end -->

## 两条路线都可能给 KKT 必要性
<!-- bilingual-en:start -->
*Both routes can support KKT necessity for different reasons*
<!-- bilingual-en:end -->

LICQ 通过候选点的局部法向几何支持 [[KKT必要性]]；Slater 通过 [[Slater强对偶|强对偶与乘子存在]] 支持凸问题的 KKT 必要方向。相同的下游结论不把上游假设变成同义词。

<!-- bilingual-en:start -->
LICQ supports KKT necessity through local normal geometry, while Slater supports it in convex problems through strong duality and multiplier existence. Sharing a downstream conclusion does not make the assumptions synonymous.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么复制一条活动约束可能破坏 LICQ，却不改变 Slater？
>
> <!-- bilingual-en:start -->
> Why can duplicating an active constraint destroy LICQ without changing Slater's condition?
> <!-- bilingual-en:end -->
>
> **答案：** 复制会制造线性相关的活动梯度；但可行集和严格可行点没有改变。
>
> <!-- bilingual-en:start -->
> **Answer:** Duplication creates linearly dependent active gradients, but it leaves the feasible set and every strictly feasible point unchanged.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Boyd and Vandenberghe, *Convex Optimization*, §5.2.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)：核对 Slater 的凸问题严格可行性定义。
- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：核对 LICQ 的局部活动梯度定义。

<!-- bilingual-en:start -->
- Boyd and Vandenberghe, *Convex Optimization*, §5.2.3 was checked for Slater's strict-feasibility definition.
- MIT 6.7220 Lecture 7 was checked for LICQ as a local active-gradient condition.
<!-- bilingual-en:end -->
