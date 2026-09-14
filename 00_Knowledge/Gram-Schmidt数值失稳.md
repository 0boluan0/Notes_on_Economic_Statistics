---
aliases:
  - "经典 Gram-Schmidt 在近线性相关列上可能显著损失正交性"
  - Classical Gram-Schmidt can lose orthogonality
  - CGS loss of orthogonality
student_os: knowledge-atom
atom_id: LA-PROJ-009
atom_set: orthogonal-projection-least-squares
atom_type: numerical-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Gram-Schmidt正交化]]"
related:
  - "[[薄QR最小二乘]]"
  - "[[正规方程条件数平方]]"
  - "[[条件性与算法稳定性]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---

# 经典 Gram-Schmidt 在近线性相关列上可能显著损失正交性
<!-- bilingual-en:start -->
*Classical Gram-Schmidt can lose substantial orthogonality when the input columns are nearly linearly dependent*
<!-- bilingual-en:end -->

> [!summary] 数值边界
> 在精确算术中，经典 [[Gram-Schmidt正交化]] 得到 $A=QR$ 且 $Q^TQ=I$。在浮点运算中，当某列几乎落在前面列的张成空间时，计算出的 $Q$ 可能明显偏离正交。因此 $A\approx QR$ 的分解残差很小，不等于 $Q^TQ\approx I$。
> <!-- bilingual-en:start -->
> Classical Gram-Schmidt gives $A=QR$ and $Q^TQ=I$ in exact arithmetic. In floating-point arithmetic, a column that nearly lies in the span of earlier columns can produce a computed $Q$ that is far from orthogonal. A small factorization residual $A-QR$ therefore does not imply that $Q^TQ$ is close to the identity.
> <!-- bilingual-en:end -->

## 失稳从哪里来
<!-- bilingual-en:start -->
*Where the loss comes from*
<!-- bilingual-en:end -->

若 $a_k$ 几乎落在已有方向的张成空间中，真实余量
$$
v_k=a_k-\sum_{i<k}q_i(q_i^Ta_k)
$$
很小，却由几个较大的近似量相减得到。舍入误差相对于 $v_k$ 被放大，归一化后便可能重新出现本应消去的方向分量。
<!-- bilingual-en:start -->
When $a_k$ nearly belongs to the span of earlier directions, the true remainder is small but is formed by subtracting larger approximate quantities. Rounding error can dominate that small remainder, and normalization can reintroduce components that exact arithmetic would have removed.
<!-- bilingual-en:end -->

## 怎样选实现
<!-- bilingual-en:start -->
*Choosing an implementation*
<!-- bilingual-en:end -->

modified Gram-Schmidt 通常比经典形式更能保持正交性，但在高度病态的输入上仍可能需要再正交化。对稠密矩阵的通用 QR 分解，Householder 反射通常是更稳健的默认方法。这些是实现层选择，不改变精确的 [[QR分解]] 定义。
<!-- bilingual-en:start -->
Modified Gram-Schmidt usually preserves orthogonality better than the classical form, although severely ill-conditioned inputs may still require reorthogonalization. Householder reflectors are the usual robust default for dense QR factorization. These are implementation choices; they do not change the exact definition of a [[QR分解|QR factorization]].
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> CGS 给出很小的 $\|A-QR\|$，是否已经证明 $Q$ 近似正交？
> <!-- bilingual-en:start -->
> If CGS gives a small $\|A-QR\|$, does that alone show that $Q$ is nearly orthogonal?
> <!-- bilingual-en:end -->
>
> **答案：** 没有。还必须单独检查 $\|Q^TQ-I\|$；分解残差与正交性误差是两个不同的数值指标。
> <!-- bilingual-en:start -->
> **Answer:** No. One must also check $\|Q^TQ-I\|$; factorization residual and loss of orthogonality are distinct numerical diagnostics.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT 18.335J Week 4](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/week-4/)：核验 CGS 在近线性相关列上损失正交性，以及 modified Gram-Schmidt 和 Householder QR 的实现边界。
- Giraud, Langou and Rozložník, [“The Loss of Orthogonality in the Gram–Schmidt Orthogonalization Process”](https://doi.org/10.1016/j.camwa.2005.08.009), 2005：核验正交性损失与条件性、改进形式及再正交化的关系。
<!-- bilingual-en:start -->
- MIT 18.335J Week 4 supports loss of orthogonality for nearly dependent columns and the implementation boundaries among classical Gram-Schmidt, modified Gram-Schmidt, and Householder QR.
- Giraud, Langou, and Rozložník (2005) support the conditioning-dependent loss of orthogonality and the role of reorthogonalization.
<!-- bilingual-en:end -->
