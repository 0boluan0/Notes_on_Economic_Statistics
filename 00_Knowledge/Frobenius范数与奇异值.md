---
aliases:
  - "矩阵的 Frobenius 范数平方等于全部奇异值平方之和"
  - Frobenius norm from singular values
student_os: knowledge-atom
atom_id: LA-SVD-008
atom_set: singular-value-decomposition-low-rank
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[奇异值]]"
  - "[[Frobenius范数]]"
  - "[[迹与特征值总和]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[奇异值能量比]]"
  - "[[截断SVD最佳低秩近似]]"
related:
  - "[[谱范数与最大奇异值]]"
---

# 矩阵的 Frobenius 范数平方等于全部奇异值平方之和
<!-- bilingual-en:start -->
*The squared Frobenius norm of a matrix equals the sum of the squares of all its singular values*
<!-- bilingual-en:end -->

> [!summary] 核心恒等式
> 对任意实矩阵或复矩阵 $A$，
> $$
> \|A\|_F^2
> =\operatorname{tr}(A^*A)
> =\sum_i\sigma_i^2.
> $$
> 因而 Frobenius 范数汇总矩阵在全部奇异方向上的平方伸缩，而不是只取最大方向。
> <!-- bilingual-en:start -->
> The Frobenius norm aggregates squared stretch across every singular direction, and equals the sum of squared matrix entries.
> <!-- bilingual-en:end -->

由 [[Frobenius范数]] 的定义，
$$
\|A\|_F^2=\sum_{j,k}|a_{jk}|^2.
$$
$A^*A$ 的第 $j$ 个对角元是 $A$ 第 $j$ 列元素模平方之和，所以这个总和等于 $\operatorname{tr}(A^*A)$。再由 [[迹与特征值总和]] 与 [[奇异值与Gram谱|Gram 谱]]，迹等于全部 $\sigma_i^2$ 之和。

例如 $A=\operatorname{diag}(4,3,0)$ 时，
$$
\|A\|_F=\sqrt{4^2+3^2}=5.
$$
这里恰好得到 5 只是勾股关系；它与谱范数 $4$ 回答的是不同问题。

> [!question]- 自检
> 非零奇异值为 $5,2,2$ 时，Frobenius 范数是多少？
>
> **答案：** $\sqrt{25+4+4}=\sqrt{33}$。

## 来源与核验

- [Cornell CS 6241, SVD and low-rank approximation](https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-02-13.html)：核对 Frobenius 范数、迹与奇异值平方和恒等式。
- [LAPACK Users' Guide: Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html)：核对 Gram 谱与奇异值的关系。
