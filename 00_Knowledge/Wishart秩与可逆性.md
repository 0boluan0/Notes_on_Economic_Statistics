---
aliases:
  - "正定尺度和整数自由度下 Wishart 矩阵的秩几乎必然等于维数与自由度的较小者"
  - With a positive-definite scale and integer degrees of freedom a Wishart matrix has almost-sure rank equal to the smaller of dimension and degrees of freedom
  - Wishart rank and invertibility
  - Wishart 秩与可逆性
student_os: knowledge-atom
atom_id: STAT-WISH-004
atom_set: wishart-sample-covariance
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Wishart分布]]"
  - "[[正定与半正定可逆性]]"
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
related:
  - "[[样本协方差Wishart律]]"
---

# 正定尺度和整数自由度下 Wishart 矩阵的秩几乎必然等于维数与自由度的较小者
<!-- bilingual-en:start -->
*With a positive-definite scale matrix and integer degrees of freedom, a Wishart matrix almost surely has rank equal to the smaller of its dimension and degrees of freedom*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 若 $\nu$ 是正整数，$W\sim W_p(\Sigma,\nu)$ 由 $\nu$ 个独立 $N_p(0,\Sigma)$ 向量的外积和生成，且 $\Sigma\succ0$，则
> $$\operatorname{rank}(W)=\min(p,\nu)\qquad\text{几乎必然}.$$
> 因而 $W$ 几乎必然正定且可逆，当且仅当 $\nu\ge p$。
> <!-- bilingual-en:start -->
> For an integer-degree Wishart matrix with positive-definite scale, the rank is almost surely $\min(p,\nu)$; invertibility therefore requires and, almost surely, follows from $\nu\ge p$.
> <!-- bilingual-en:end -->

把列向量排成 $p\times\nu$ 矩阵 $Z=[Z_1\ \cdots\ Z_\nu]$，便有 $W=ZZ^T$，所以
$$
\operatorname{rank}(W)=\operatorname{rank}(Z)\le\min(p,\nu).
$$
由于 $\Sigma\succ0$，$Z$ 具有相对于欧氏空间 Lebesgue 测度的连续密度；所有最大阶子式同时为零只占零测集，所以 $Z$ 几乎必然达到最大可能秩。这既解释了秩上界，也解释了为什么等号不是额外假设。

对总体协方差 $\Sigma\succ0$ 的 iid 多元正态样本，$(n-1)S\sim W_p(\Sigma,n-1)$，故
$$
\operatorname{rank}(S)=\min(p,n-1)\qquad\text{几乎必然}.
$$
因此普通逆 $S^{-1}$ 几乎必然存在的门槛是 $n-1\ge p$，即 $n\ge p+1$。这只是**代数可逆**门槛；当 $p$ 已非常接近 $n$ 时，最小特征值仍可能很小，求逆会放大估计误差。可逆不等于数值稳定，也不等于推断可靠。

> [!warning] 边界
> 若 $\Sigma$ 本身奇异且秩为 $r$，$W$ 的列空间被限制在 $\operatorname{range}(\Sigma)$ 内，并且在同一个整数自由度生成式下
> $$\operatorname{rank}(W)=\min(r,\nu)\qquad\text{几乎必然}.$$
> 另一个不能混用的边界是：非奇异 Wishart 密度允许实数 $\nu>p-1$，并直接以 $W\succ0$ 为支持。例如 $p-1<\nu<p$ 时它仍几乎必然满秩；非整数 $\nu$ 不表示向量个数，不能把本卡的“$\nu\ge p$”整数门槛机械套过去。样本协方差的自由度 $n-1$ 本来就是整数，所以不受这个区别影响。

> [!question]- 自检
> 为什么 $n=p$ 时，正态样本的 $p\times p$ 样本协方差仍必然奇异？
>
> **答案：** 中心化消耗一个样本方向，故 $S$ 的秩至多 $n-1=p-1<p$。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|多元统计课程 §1.6]]：核对样本协方差的 Wishart 自由度。
- [[正定与半正定可逆性]]：核对正定、半正定与可逆性的边界；秩结论由 $W=ZZ^T$ 直接推出。
- [Purdue Statistics, *Graduate Probability*, Definition 1.76 and Theorem 1.91](https://www.stat.purdue.edu/~dasgupta/gradprob.pdf#page=171)：核对正定 Wishart 密度的自由度范围及正态样本协方差几乎必然正定的样本量门槛。
- [Stan Functions Reference, Wishart distribution](https://mc-stan.org/docs/functions-reference/covariance_matrix_distributions.html#wishart-distribution)：核对非奇异密度允许实数 $\nu>p-1$，且支持为正定矩阵。
