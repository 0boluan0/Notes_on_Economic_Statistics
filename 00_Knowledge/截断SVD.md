---
aliases:
  - "秩 k 截断 SVD 保留一个 SVD 的前 k 个奇异分量并舍去其余分量"
  - 截断奇异值分解
  - Rank-k truncated SVD
student_os: knowledge-atom
atom_id: LA-SVD-019
atom_set: singular-value-decomposition-low-rank
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[紧SVD]]"
  - "[[奇异向量]]"
  - "[[SVD秩一展开]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[截断SVD最佳低秩近似]]"
  - "[[奇异值能量比]]"
related:
  - "[[低秩近似]]"
  - "[[Frobenius低秩近似唯一性]]"
  - "[[谱范数低秩近似不唯一]]"
---

# 秩 k 截断 SVD 保留一个 SVD 的前 k 个奇异分量并舍去其余分量
<!-- bilingual-en:start -->
*A rank-k truncated SVD keeps the first k singular components of an SVD and discards the rest*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 若 $A\in\mathbb F^{m\times n}$ 的秩为 $r$，紧 SVD 写成
> $$
> A=\sum_{i=1}^{r}\sigma_i u_iv_i^*,
> \qquad \sigma_1\ge\cdots\ge\sigma_r>0.
> $$
> 对整数 $0\le k\le r$，秩 $k$ 截断 SVD 定义为
> $$
> A_k=\sum_{i=1}^{k}\sigma_i u_iv_i^*.
> $$
> 它保留最大的 $k$ 个奇异分量，把其余分量置零。
> <!-- bilingual-en:start -->
> The rank-$k$ truncation retains the first $k$ singular components and sets all remaining components to zero.
> <!-- bilingual-en:end -->

因为每个保留项对应一个彼此正交的正奇异方向，所以 $\operatorname{rank}(A_k)=k$，其中 $A_0=0$。当 $k=r$ 时，$A_r=A$，只是换了一种写法；只有 $k<r$ 时，截断才改变矩阵并产生近似误差。

例如
$$
A=\operatorname{diag}(5,2,1)
$$
的秩 $2$ 截断是 $A_2=\operatorname{diag}(5,2,0)$。这一步只定义了怎样截断；它为什么在常用范数下最佳，要由 [[截断SVD最佳低秩近似]] 说明。

当 $1\le k<r$ 且 $\sigma_k=\sigma_{k+1}$ 时，截断位置切穿一个重奇异子空间。不同的标准正交基可以给出不同的 $A_k$，所以“前 $k$ 个方向”并非由矩阵唯一指定。Frobenius 与谱范数下的最优解各有不同的唯一性结论，见 [[Frobenius低秩近似唯一性]] 与 [[谱范数低秩近似不唯一]]。

> [!question]- 自检
> 一个秩为 $4$ 的矩阵取 $k=4$ 时，截断 SVD 是否还是近似？
>
> **答案：** $A_4=A$，重构误差为零；它是完整紧 SVD 的秩一展开，并未丢弃任何正奇异分量。

## 来源与核验

- MIT OpenCourseWare 18.065, [Lecture 7: Eckart–Young](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-7-eckart-young-the-closest-rank-k-matrix-to-a/)：核对秩 $k$ 截断的构造。
- [Netlib, Randomized Numerical Linear Algebra: Foundations and Algorithms](https://www.netlib.org/utk/people/JackDongarra/PAPERS/rand-templates.pdf), §4.1.1：核对截断 SVD、秩与尾部误差的表述。
