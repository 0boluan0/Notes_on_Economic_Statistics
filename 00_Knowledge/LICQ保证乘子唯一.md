---
aliases:
  - '在 KKT 乘子存在时，活动约束梯度满足 LICQ 会使该点的不等式与等式乘子唯一'
  - When KKT multipliers exist LICQ makes the inequality and equality multipliers at that point unique
student_os: knowledge-atom
atom_id: OPT-MV-020
atom_set: multivariable-optimization
atom_type: uniqueness-theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[LICQ条件]]"
  - "[[KKT条件]]"
related:
  - "[[严格凹不保证乘子唯一]]"
  - "[[绑定不推正乘子]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# 在 KKT 乘子存在时，活动约束梯度满足 LICQ 会使该点的不等式与等式乘子唯一
<!-- bilingual-en:start -->
*When KKT multipliers exist LICQ makes the inequality and equality multipliers at that point unique*
<!-- bilingual-en:end -->

> [!summary] 唯一性来自法向量表示唯一
> 设 $x^*$ 已有 KKT 乘子。若等式约束梯度与所有活动不等式约束梯度满足 [[LICQ条件|LICQ]]，则这些乘子唯一。非活动不等式的乘子由互补松弛固定为零；活动约束与等式约束的乘子则是把 $\nabla f(x^*)$ 表示成一组线性无关法向量的系数。
>
> <!-- bilingual-en:start -->
> Suppose KKT multipliers exist at $x^*$. Under LICQ, equality gradients and active inequality gradients are linearly independent. Inactive multipliers are zero by complementary slackness, while the remaining multipliers are the unique coefficients representing the objective gradient in that independent family of normals.
> <!-- bilingual-en:end -->

## 一行证明
<!-- bilingual-en:start -->
*One-line proof*
<!-- bilingual-en:end -->

若 $(\lambda,\nu)$ 与 $(\tilde\lambda,\tilde\nu)$ 都满足同一点的驻点方程，两式相减得到
$$
\sum_{i\in\mathcal A(x^*)}(\lambda_i-\tilde\lambda_i)\nabla g_i(x^*)
+\sum_j(\nu_j-\tilde\nu_j)\nabla h_j(x^*)=0.
$$
LICQ 使这组梯度线性无关，因此每个系数差都为零。再结合非活动乘子全为零，得到完整乘子向量相同。

<!-- bilingual-en:start -->
Subtracting two stationarity equations gives a zero linear combination of equality and active-inequality gradients. LICQ forces every coefficient difference to vanish. Complementary slackness already fixes inactive multipliers at zero, so the complete multiplier vector is unique.
<!-- bilingual-en:end -->

## 它没有保证什么
<!-- bilingual-en:start -->
*What this theorem does not guarantee*
<!-- bilingual-en:end -->

这条唯一性结论以“乘子已经存在”为前提；乘子存在的条件见 [[KKT必要性]]。乘子唯一也不证明 $x^*$ 最优，更不证明最优选择唯一。反过来，目标严格凹可能使选择唯一，却仍不能修复相关或重复约束造成的乘子不唯一，见 [[严格凹不保证乘子唯一]]。

<!-- bilingual-en:start -->
This result assumes multiplier existence. It proves neither optimality nor uniqueness of the primal choice. Conversely, strict concavity may make the choice unique while dependent or duplicated constraints still leave multipliers nonunique.
<!-- bilingual-en:end -->

> [!question]- 自检
> LICQ 为什么能排除同一目标梯度的两组不同乘子表示？
>
> <!-- bilingual-en:start -->
> Why does LICQ rule out two different multiplier representations of the same objective gradient?
> <!-- bilingual-en:end -->
>
> **答案：** 两组表示相减会给出活动法向量的零线性组合；LICQ 的线性无关迫使所有系数差为零。
>
> <!-- bilingual-en:start -->
> **Answer:** Subtracting the two representations gives a zero linear combination of active normals. LICQ forces every coefficient difference to vanish.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：核对 LICQ、活动梯度与乘子唯一性的关系。
- [Stanford CME307/MS&E311, Lecture Note 7](https://web.stanford.edu/class/msande311/lecture07.pdf)：交叉核对正则点的法向量表示。

<!-- bilingual-en:start -->
- MIT 6.7220 Lecture 7 was checked for LICQ and multiplier uniqueness.
- Stanford CME307/MS&E311 Lecture Note 7 was used to cross-check the normal-vector representation at a regular point.
<!-- bilingual-en:end -->
