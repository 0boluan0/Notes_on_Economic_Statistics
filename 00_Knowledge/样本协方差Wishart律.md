---
aliases:
  - "正态样本的 n-1 倍样本协方差服从自由度 n-1 的 Wishart 分布"
  - The scaled sample covariance of a normal sample is Wishart
  - Wishart law of sample covariance
  - 样本协方差的 Wishart 分布
student_os: knowledge-atom
atom_id: STAT-WISH-002
atom_set: wishart-sample-covariance
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Wishart分布]]"
  - "[[样本协方差矩阵]]"
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
related:
  - "[[正态均值协方差独立]]"
  - "[[Wishart抽样假设]]"
---

# 正态样本的 n-1 倍样本协方差服从自由度 n-1 的 Wishart 分布
<!-- bilingual-en:start -->
*For a normal sample, $(n-1)$ times the sample covariance has a Wishart distribution with $n-1$ degrees of freedom*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 若 $n\ge2$，$X_1,\ldots,X_n\overset{iid}{\sim}N_p(\mu,\Sigma)$，并定义
> $$\bar X=\frac1n\sum_{i=1}^nX_i,\qquad
> S=\frac1{n-1}\sum_{i=1}^n(X_i-\bar X)(X_i-\bar X)^T,$$
> 则在 scale 约定下
> $$(n-1)S\sim W_p(\Sigma,n-1).$$
> <!-- bilingual-en:start -->
> Centring an iid multivariate-normal sample leaves $n-1$ independent residual directions, giving the displayed Wishart law.
> <!-- bilingual-en:end -->

自由度少 1 不是由分母记号机械造成，而是中心化残差满足
$$
\sum_{i=1}^n(X_i-\bar X)=0,
$$
所以 $n$ 个残差中只有 $n-1$ 个独立样本方向。通过观测索引空间中的正交变换，可把其中一维变成 $\sqrt n\,\bar X$，其余 $n-1$ 维变成相互独立的中心 Gaussian 对比；这些对比的外积和正是 $(n-1)S$。

若改用 MLE 协方差 $\widehat\Sigma_{\rm MLE}=n^{-1}\sum(X_i-\bar X)(X_i-\bar X)^T$，则 $n\widehat\Sigma_{\rm MLE}$ 服从同一个 $W_p(\Sigma,n-1)$。先确认分母，才能正确缩放。

当 $n-1<p$ 时，这里的 $W_p(\Sigma,n-1)$ 指外积生成的**奇异 Wishart 律**；它仍是正确的矩阵分布等式，但没有正定矩阵空间上的普通密度。有些只定义非奇异 Wishart 的教材或软件会拒绝这组参数，不能据此把自由度改成 $n$。

> [!question]- 自检
> 为什么把样本协方差分母从 $n-1$ 改成 $n$ 不会把 Wishart 自由度改成 $n$？
>
> **答案：** 自由度来自中心化后的独立残差方向数，仍是 $n-1$；改分母只改变矩阵前的缩放常数。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|多元统计课程 §§1.5.2–1.6.1]]：核对 $S$ 的分母、Wishart 缩放与自由度。
- [Penn State STAT 505, Lesson 4](https://online.stat.psu.edu/stat505/Lesson04)：核对多元正态抽样与样本协方差的课程语境。
- [Stanford STATS 305C, *One sample problem*](https://web.stanford.edu/class/stats305c/lectures/Onesample.html#wishart-distribution)：核对 $n\widehat\Sigma_{\rm MLE}$ 与 $(n-1)S$ 具有同一自由度 $n-1$ 的 Wishart 律。
