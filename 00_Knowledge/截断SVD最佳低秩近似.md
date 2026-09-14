---
aliases:
  - "截断 SVD 在谱范数和 Frobenius 范数下达到最佳低秩近似误差"
  - 截断 SVD 最佳低秩近似
  - Best low-rank approximation by truncated SVD
  - Eckart–Young–Mirsky theorem
student_os: knowledge-atom
atom_id: LA-SVD-003
atom_set: singular-value-decomposition-low-rank
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[低秩近似]]"
  - "[[截断SVD]]"
  - "[[SVD秩一展开]]"
  - "[[谱范数]]"
  - "[[谱范数与最大奇异值]]"
  - "[[Frobenius范数]]"
  - "[[Frobenius范数与奇异值]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[Frobenius低秩近似唯一性]]"
  - "[[谱范数低秩近似不唯一]]"
related:
  - "[[奇异值能量比]]"
  - "[[截断SVD与岭正则化]]"
---

# 截断 SVD 在谱范数和 Frobenius 范数下达到最佳低秩近似误差
<!-- bilingual-en:start -->
*The truncated SVD attains the best low-rank approximation error in both the spectral and Frobenius norms*
<!-- bilingual-en:end -->

> [!summary] 核心定理
> 设 $A\in\mathbb F^{m\times n}$，其中 $\mathbb F=\mathbb R$ 或 $\mathbb C$，且 $\operatorname{rank}(A)=r$。写出紧 SVD
> $$
> A=\sum_{i=1}^{r}\sigma_i u_iv_i^*,
> \qquad \sigma_1\ge\cdots\ge\sigma_r>0.
> $$
> 对 $0\le k<r$，令 $A_0=0$，并令 $A_k=\sum_{i=1}^{k}\sigma_i u_iv_i^*$。在全部满足 $B\in\mathbb F^{m\times n}$ 且 $\operatorname{rank}(B)\le k$ 的矩阵中，$A_k$ 同时达到
> $$
> \min_{\operatorname{rank}(B)\le k}\|A-B\|_2=\sigma_{k+1},
> \qquad
> \min_{\operatorname{rank}(B)\le k}\|A-B\|_F=\left(\sum_{i=k+1}^{r}\sigma_i^2\right)^{1/2}.
> $$
> <!-- bilingual-en:start -->
> The rank-$k$ truncation minimises both spectral and Frobenius reconstruction error, with errors determined by the discarded singular values.
> <!-- bilingual-en:end -->

$A_k$ 保留前 $k$ 个正交秩一分量，残差只含尾部。它的最大伸缩是 $\sigma_{k+1}$，平方总量是尾部奇异值平方和。Eckart–Young–Mirsky 定理进一步保证，没有其他秩不超过 $k$ 的矩阵能把相应范数降得更低。

谱范数下界可直接看出：任意秩至多 $k$ 的 $B$，其零空间必与 $\operatorname{span}(v_1,\ldots,v_{k+1})$ 有非零交；在该交中取单位向量可得 $\|A-B\|_2\ge\sigma_{k+1}$。Frobenius 的尾部平方和下界是定理对酉不变范数的更强结论，不能由“至少漏掉一个方向”单独推出。

这里的“最佳”只相对于指定范数与秩约束。Frobenius 范数下的唯一性由截断处的奇异值间隔决定，见 [[Frobenius低秩近似唯一性]]；非平凡谱范数问题则总有多个最优解，见 [[谱范数低秩近似不唯一]]。重构误差很小也不能保证被删方向与下游任务无关，见 [[奇异值能量不等于任务信息]]。

> [!question]- 自检
> 奇异值为 $9,4,4,1$，取 $k=2$ 时，最佳谱误差和 Frobenius 误差各是多少？
>
> **答案：** $4$ 与 $\sqrt{4^2+1^2}=\sqrt{17}$。

## 来源与核验

- C. Eckart and G. Young (1936), [*The Approximation of One Matrix by Another of Lower Rank*](https://doi.org/10.1007/BF02288367)：原始最佳低秩近似结果。
- [Netlib, Randomized Numerical Linear Algebra: Foundations and Algorithms](https://www.netlib.org/utk/people/JackDongarra/PAPERS/rand-templates.pdf), §4.1.1：核对两种误差公式与酉不变范数表述。
- MIT OpenCourseWare 18.065, [Lecture 7: Eckart–Young](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-7-eckart-young-the-closest-rank-k-matrix-to-a/)：核对截断 SVD 的最佳秩-$k$ 解释。
