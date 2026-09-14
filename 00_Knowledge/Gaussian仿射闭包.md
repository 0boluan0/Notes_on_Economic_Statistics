---
aliases:
  - "联合 Gaussian 在仿射变换与取边际下仍为 Gaussian"
  - "Affine closure and marginals of a multivariate Gaussian"
  - "多元正态的仿射变换"
student_os: knowledge-atom
atom_id: STAT-MVN-002
atom_set: multivariate-normal
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[联合Gaussian]]"
part_of:
  - "[[多元正态分布.canvas]]"
related:
  - "[[Gaussian条件分布]]"
  - "[[联合高斯独立判据]]"
---

# 联合 Gaussian 在仿射变换与取边际下仍为 Gaussian
<!-- bilingual-en:start -->
*A jointly Gaussian vector remains Gaussian under affine transformations and marginalisation*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 若 $X\sim N_p(\mu,\Sigma)$，$A\in\mathbb R^{q\times p}$ 与 $b\in\mathbb R^q$ 都是确定的，则
> $$AX+b\sim N_q(A\mu+b,\ A\Sigma A^T).$$
> $A$ 不必为方阵、满秩或可逆；秩不足只会使结果的协方差奇异。
> <!-- bilingual-en:start -->
> Every deterministic affine image of a jointly Gaussian vector is jointly Gaussian, with the transformed mean and covariance shown above. Rank deficiency is allowed and may produce a degenerate law.
> <!-- bilingual-en:end -->

由 $X=\mu+BZ$ 可直接看出
$$
AX+b=(A\mu+b)+(AB)Z,
$$
所以结论不需要 Jacobian，也不要求原分布或像分布有全维密度。选取只保留若干坐标的 selection matrix $P$，就得到边际分布
$$
PX\sim N(P\mu,P\Sigma P^T).
$$
因此任何分量子向量仍联合 Gaussian。
<!-- bilingual-en:start -->
The constructive representation proves the result even in singular cases. Marginalisation is the special case in which the matrix selects a subset of coordinates.
<!-- bilingual-en:end -->

前提是整个 $X$ **联合** Gaussian。若 $X_1,X_2$ 仅分别正态，$X_1+X_2$ 不一定正态。结论也只对仿射变换成立；例如非退化标准正态的平方 $X_1^2$ 服从卡方而不是正态。

> [!question]- 自检
> 若 $A$ 的两行完全相同，为什么仍可使用公式，但 $AX$ 的协方差一定奇异？
>
> **答案：** 公式允许任意确定矩阵；两行相同使 $AX$ 的两个分量几乎处处相等，产生零方差的线性方向，所以 $A\Sigma A^T$ 奇异。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.3.2. 仿射变换|多元统计课程 §1.3.2]]：核对仿射变换的均值与协方差公式。
- [MIT OCW 6.436J, Lecture 14, Theorem 1(c)](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf)：核对任意矩形矩阵下的仿射闭包和退化边界。
- [MIT OCW 18.443, Lecture 4](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/da928274a46632ce0d35d2012a1c8e74_lecture4.pdf)：核对 $A\Sigma A^T$ 公式及非满秩变换仍定义 Gaussian 向量。
