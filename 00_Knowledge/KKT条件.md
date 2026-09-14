---
aliases:
  - 'KKT 条件是原问题可行性、对偶可行性、驻点和互补松弛共同组成的联立系统'
  - The KKT conditions are the joint system of primal feasibility dual feasibility stationarity and complementary slackness
student_os: knowledge-atom
atom_id: OPT-MV-005
atom_set: multivariable-optimization
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Lagrange必要条件]]"
related:
  - "[[LICQ条件]]"
  - "[[Slater条件]]"
leads_to:
  - "[[互补松弛]]"
  - "[[KKT必要性]]"
  - "[[KKT充分性]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# KKT 条件是原问题可行性、对偶可行性、驻点和互补松弛共同组成的联立系统
<!-- bilingual-en:start -->
*The KKT conditions are the joint system of primal feasibility dual feasibility stationarity and complementary slackness*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 考虑
> $$
> \max_x f(x)\quad\text{s.t.}\quad h(x)=0,\qquad g(x)\le0,
> $$
> 并统一取
> $$
> \mathcal L(x,\lambda,\nu)=f(x)-\lambda^Tg(x)-\nu^Th(x).
> $$
> 对一个候选三元组 $(x^*,\lambda^*,\nu^*)$，Karush–Kuhn–Tucker（KKT）条件就是下面四类等式与不等式同时成立。它先定义一个候选系统；局部最优何时必须进入这个系统由 [[KKT必要性]] 负责，系统何时足以证明全局最优由 [[KKT充分性]] 负责。
>
> <!-- bilingual-en:start -->
> For the stated maximisation problem and sign convention, the KKT conditions are the following four classes imposed jointly on a candidate triple. These equations define the system. [[KKT必要性|KKT necessity]] states when a local optimum must enter it, while [[KKT充分性|KKT sufficiency]] states when the system certifies global optimality.
> <!-- bilingual-en:end -->

## 四类条件
<!-- bilingual-en:start -->
*The four classes of conditions*
<!-- bilingual-en:end -->

**原问题可行性：**
$$
h(x^*)=0,
\qquad g(x^*)\le0.
$$
候选点必须真的在可行域内。

<!-- bilingual-en:start -->
**Primal feasibility:** $h(x^*)=0$ and $g(x^*)\le0$. The candidate must actually belong to the feasible set.
<!-- bilingual-en:end -->

**对偶可行性：**
$$
\lambda^*\ge0.
$$
这个符号与“最大化、$g\le0$、$\mathcal L=f-\lambda^Tg-\nu^Th$”的统一约定配套。等式乘子 $\nu$ 没有符号限制。

<!-- bilingual-en:start -->
**Dual feasibility:** $\lambda^*\ge0$. This sign belongs to the chosen convention: maximisation, $g\le0$, and $\mathcal L=f-\lambda^Tg-\nu^Th$. Equality multipliers $\nu$ are unrestricted in sign.
<!-- bilingual-en:end -->

**驻点：**
$$
\nabla f(x^*)-Dg(x^*)^T\lambda^*-Dh(x^*)^T\nu^*=0.
$$
目标的一阶改善方向被活动约束的法向量抵消。

<!-- bilingual-en:start -->
**Stationarity:** $\nabla f(x^*)-Dg(x^*)^T\lambda^*-Dh(x^*)^T\nu^*=0$. The objective's first-order improvement is balanced by normals to the active constraints.
<!-- bilingual-en:end -->

**互补松弛：**
$$
\lambda_i^*g_i(x^*)=0\qquad(i=1,\ldots,m).
$$
每条不等式的乘子与约束值乘积必须为零。怎样读取这项定义，以及哪些反向推论不成立，由 [[互补松弛]] 展开。

<!-- bilingual-en:start -->
**Complementary slackness:** $\lambda_i^*g_i(x^*)=0$ for every inequality. [[互补松弛|The complementary-slackness explanation]] shows how to read this definition and which converses fail.
<!-- bilingual-en:end -->

## 使用顺序与逻辑边界
<!-- bilingual-en:start -->
*Order of use and logical boundary*
<!-- bilingual-en:end -->

先写清目标方向、约束方向和 Lagrangian 号约定，再逐条检查四类条件。把所有不等式预先当成等式会丢掉内部解；只解驻点方程而不检查 $g\le0$ 与 $\lambda\ge0$ 会留下不可行或符号错误的伪解。

<!-- bilingual-en:start -->
State the objective direction, inequality direction, and Lagrangian sign convention before checking all four classes. Treating every inequality as an equality discards interior solutions. Solving stationarity without checking $g\le0$ and $\lambda\ge0$ leaves infeasible or sign-inconsistent false candidates.
<!-- bilingual-en:end -->

解出 KKT 系统只得到候选三元组。[[KKT必要性]] 需要可微性与适当约束资格；[[KKT充分性]] 需要凹目标、凸不等式与仿射等式等结构。不要把这些定理的假设塞回 KKT 的定义，也不要只解驻点方程就忽略另外三类条件。

<!-- bilingual-en:start -->
Solving the KKT system produces a candidate triple. [[KKT必要性|Necessity]] needs differentiability and a suitable constraint qualification; [[KKT充分性|sufficiency]] needs convex structure. Those theorem assumptions are not part of the definition itself.
<!-- bilingual-en:end -->

> [!question]- 自检
> 已经解出驻点方程后，为什么仍必须把 $x^*$ 和乘子代回另外三类条件？
>
> <!-- bilingual-en:start -->
> After solving stationarity, why must the candidate and multipliers still be substituted into the other three classes of conditions?
> <!-- bilingual-en:end -->
>
> **答案：** 驻点方程本身不保证原问题可行、乘子符号正确或松弛约束的乘子为零；KKT 的信息来自四类条件的联立。
>
> <!-- bilingual-en:start -->
> **Answer:** Stationarity alone does not guarantee primal feasibility, correct multiplier signs, or a zero multiplier on a slack constraint. KKT obtains its force from the joint system.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*SOFP Lecture Notes*（课程讲义）：核对课程采用的最大化问题、$g\le0$ 方向、乘子符号与四类条件。
- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：交叉核对 KKT 四类条件的标准系统；必要性定理由 [[KKT必要性]] 独立承担。
- [Boyd and Vandenberghe, *Convex Optimization*, §5.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)：核对 KKT 系统及其与凸充分性的逻辑分界。

<!-- bilingual-en:start -->
- The EC400 SOFP lecture notes were checked for the course convention: maximisation, $g\le0$, multiplier signs, and the four KKT classes.
- MIT 6.7220 Lecture 7 was used to cross-check the standard four-part KKT system; the necessity theorem is stated in [[KKT必要性|KKT necessity]].
- Boyd and Vandenberghe, *Convex Optimization*, §5.5 was checked for the KKT system and the logical boundary between necessity and convex sufficiency.
<!-- bilingual-en:end -->
