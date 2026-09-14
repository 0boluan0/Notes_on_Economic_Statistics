---
aliases:
  - '在凸问题中，Slater 条件在通常的有限最优值假设下保证强对偶与最优乘子存在，从而给出 KKT 的必要方向'
  - In a convex problem Slater's condition guarantees strong duality and optimal multipliers under the usual finite-value assumptions, yielding the necessary KKT direction
student_os: knowledge-atom
atom_id: OPT-MV-022
atom_set: multivariable-optimization
atom_type: duality-theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Slater条件]]"
  - "[[KKT条件]]"
related:
  - "[[KKT充分性]]"
leads_to:
  - "[[KKT必要性]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# 在凸问题中，Slater 条件在通常的有限最优值假设下保证强对偶与最优乘子存在，从而给出 KKT 的必要方向
<!-- bilingual-en:start -->
*In a convex problem Slater's condition guarantees strong duality and optimal multipliers under the usual finite-value assumptions, yielding the necessary KKT direction*
<!-- bilingual-en:end -->

> [!summary] 严格可行点把最优解接到对偶证书
> 对凹最大化、凸不等式与仿射等式组成的凸问题，若 [[Slater条件]] 成立，并满足通常的有限最优值假设，则原问题与对偶问题的最优值相等，而且最优对偶乘子可以取得。对可微问题，这使每个原问题最优点能够配上乘子满足 [[KKT条件]]。
>
> <!-- bilingual-en:start -->
> In a convex maximisation problem with a concave objective, convex inequalities, and affine equalities, Slater's condition gives equality of primal and dual optimal values and attainment of optimal multipliers under the usual finite-value assumptions. In the differentiable setting, a primal optimum can therefore be paired with multipliers satisfying KKT.
> <!-- bilingual-en:end -->

## 它补的是哪一个方向
<!-- bilingual-en:start -->
*Which logical direction it supplies*
<!-- bilingual-en:end -->

Slater 强对偶给出
$$
\text{凸结构 + Slater + 最优}
\Rightarrow
\text{存在 KKT 乘子}.
$$
它是一条 KKT 必要性路线。相反，若一个凸问题已经找到满足 KKT 的可行点，[[KKT充分性]] 可以直接把它认证为全局最优，不需要先验证 Slater。

<!-- bilingual-en:start -->
Slater supplies the direction from a convex optimum to the existence of KKT multipliers. The reverse certification from an existing KKT point to global optimality follows from convex structure alone and does not require Slater.
<!-- bilingual-en:end -->

## 为什么常被写成充要条件
<!-- bilingual-en:start -->
*Why it often appears in an if-and-only-if statement*
<!-- bilingual-en:end -->

把两个独立方向合在一起便得到：在可微凸问题且 Slater 成立时，$x^*$ 最优，当且仅当存在乘子使 $(x^*,\lambda^*,\nu^*)$ 满足 KKT。这里 Slater 只负责“最优推出 KKT”，不是 KKT 系统的定义，也不是“KKT 推出最优”的假设。

<!-- bilingual-en:start -->
Combining Slater-based necessity with convex KKT sufficiency yields the familiar equivalence: under Slater, a point is optimal if and only if it can be paired with KKT multipliers. Slater belongs only to the optimum-to-KKT direction.
<!-- bilingual-en:end -->

> [!question]- 自检
> 凸问题中已经找到一个满足全部 KKT 条件的点，还要验证 Slater 才能宣布全局最优吗？
>
> <!-- bilingual-en:start -->
> In a convex problem, must Slater still be verified before an existing KKT point can be declared globally optimal?
> <!-- bilingual-en:end -->
>
> **答案：** 不要。KKT 充分性不依赖 Slater；Slater 用于保证最优点拥有 KKT 乘子的反向。
>
> <!-- bilingual-en:start -->
> **Answer:** No. KKT sufficiency does not depend on Slater. Slater supplies the reverse direction that guarantees KKT multipliers at an optimum.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Boyd and Vandenberghe, *Convex Optimization*, §§5.2.3 and 5.5.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)：直接核对 Slater、强对偶、对偶最优取得与 KKT 必要/充分方向。
- [Stanford EE364A, Lecture 9 transcript](https://see.stanford.edu/materials/lsocoee364a/transcripts/ConvexOptimizationI-Lecture09.html)：交叉核对强对偶与 KKT 证书的分工。

<!-- bilingual-en:start -->
- Boyd and Vandenberghe, *Convex Optimization*, §§5.2.3 and 5.5.3 was checked for Slater, strong duality, dual attainment, and the two KKT directions.
- The Stanford EE364A Lecture 9 transcript was used to cross-check the division of roles between strong duality and KKT certification.
<!-- bilingual-en:end -->
