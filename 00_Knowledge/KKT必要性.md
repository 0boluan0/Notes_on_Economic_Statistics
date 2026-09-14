---
aliases:
  - '可微约束问题的局部最优在适当约束资格成立时存在乘子，使 KKT 四类条件同时成立'
  - At a local optimum of a differentiable constrained problem an appropriate constraint qualification guarantees multipliers satisfying all four KKT classes
student_os: knowledge-atom
atom_id: OPT-MV-018
atom_set: multivariable-optimization
atom_type: necessary-condition
status: source-checked
mastery_state: unassessed
requires:
  - "[[KKT条件]]"
  - "[[LICQ条件]]"
related:
  - "[[Slater强对偶]]"
  - "[[LICQ不是KKT必要条件]]"
  - "[[KKT充分性]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# 可微约束问题的局部最优在适当约束资格成立时存在乘子，使 KKT 四类条件同时成立
<!-- bilingual-en:start -->
*At a local optimum of a differentiable constrained problem an appropriate constraint qualification guarantees multipliers satisfying all four KKT classes*
<!-- bilingual-en:end -->

> [!summary] 这是关于局部最优的必要性定理
> 考虑
> $$
> \max_x f(x)\quad\text{s.t.}\quad h(x)=0,\qquad g(x)\le0.
> $$
> 若 $f,g,h$ 在局部最优点 $x^*$ 附近为 $C^1$，且 $x^*$ 满足适当约束资格，例如 [[LICQ条件|LICQ]]，则存在 $\lambda^*\ge0$ 与不受符号限制的 $\nu^*$，使 [[KKT条件]] 的原问题可行性、对偶可行性、驻点和互补松弛同时成立。
>
> <!-- bilingual-en:start -->
> For a differentiable problem, a local optimum satisfying an appropriate constraint qualification such as LICQ admits inequality and equality multipliers satisfying primal feasibility, dual feasibility, stationarity, and complementary slackness.
> <!-- bilingual-en:end -->

## 约束资格为什么出现在定理里
<!-- bilingual-en:start -->
*Why the theorem needs a constraint qualification*
<!-- bilingual-en:end -->

约束资格保证活动约束的一阶法向量足以描述可行域附近的局部几何。若约束表示在最优点退化，局部最优仍可能存在，却没有普通 KKT 乘子。例如最大化 $f(x)=x$、约束 $x^2\le0$ 时，唯一可行点是零，但活动约束梯度也为零，驻点方程无法用它表示 $f'(0)=1$。

<!-- bilingual-en:start -->
A constraint qualification ensures that active constraint normals adequately describe the local feasible geometry. With the degenerate constraint $x^2\le0$, zero is the only feasible point but the active gradient vanishes, so ordinary KKT stationarity cannot represent the nonzero objective derivative.
<!-- bilingual-en:end -->

## 必要不等于充分
<!-- bilingual-en:start -->
*Necessary does not mean sufficient*
<!-- bilingual-en:end -->

这条定理只说局部最优必须进入 KKT 候选集。一般非凸问题中的 KKT 点仍可能不是所求最大点。只有再加入凹目标、凸不等式与仿射等式等结构，才能用 [[KKT充分性]] 证明全局最优。LICQ 是一条方便的充分资格，却不是 KKT 乘子存在的必要条件，反例见 [[LICQ不是KKT必要条件]]。

<!-- bilingual-en:start -->
The theorem only places a local optimum inside the KKT candidate set. In a nonconvex problem, a KKT point need not be the desired maximum. Convex structure is separately required for KKT sufficiency. LICQ is a convenient sufficient qualification, not a necessary condition for KKT multipliers to exist.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若一个可微非凸问题的候选点满足全部 KKT 条件，上述必要性定理是否已证明它是局部最大？
>
> <!-- bilingual-en:start -->
> If a candidate in a differentiable nonconvex problem satisfies all KKT conditions, does the necessity theorem above prove that it is a local maximizer?
> <!-- bilingual-en:end -->
>
> **答案：** 没有。定理的方向是“合适资格下局部最优推出 KKT”；反向需要另行的充分条件。
>
> <!-- bilingual-en:start -->
> **Answer:** No. The theorem runs from a local optimum under a suitable qualification to KKT. The reverse direction needs separate sufficient conditions.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：直接核对可微约束问题中约束资格与 KKT 必要性的关系。
- LSE EC400，*SOFP Lecture Notes*（课程讲义）：核对课程的最大化号约定和四类条件。

<!-- bilingual-en:start -->
- MIT 6.7220 Lecture 7 was checked for KKT necessity under a constraint qualification.
- The EC400 SOFP notes were checked for the maximisation sign convention and the four KKT classes.
<!-- bilingual-en:end -->
