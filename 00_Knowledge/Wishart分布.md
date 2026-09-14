---
aliases:
  - "Wishart 分布在整数自由度下由独立中心 Gaussian 向量外积和生成并可由正定密度延拓到实数自由度"
  - Wishart distributions arise from Gaussian outer-product sums at integer degrees of freedom and extend through the positive-definite density
  - Wishart definition
  - Wishart 分布定义
student_os: knowledge-atom
atom_id: STAT-WISH-001
atom_set: wishart-sample-covariance
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
implies:
  - "[[Wishart秩与可逆性]]"
related:
  - "[[样本协方差Wishart律]]"
  - "[[协方差矩阵半正定性]]"
---

# Wishart 分布在整数自由度下由独立中心 Gaussian 向量外积和生成并可由正定密度延拓到实数自由度
<!-- bilingual-en:start -->
*Wishart distributions arise from Gaussian outer-product sums at integer degrees of freedom and extend to real degrees of freedom through the positive-definite density*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 在 scale 参数约定下，若正整数 $\nu$ 个向量 $Z_1,\ldots,Z_\nu$ 独立同分布为 $N_p(0,\Sigma)$，则
> $$W=\sum_{i=1}^{\nu}Z_iZ_i^T\sim W_p(\Sigma,\nu),$$
> 并且 $E[W]=\nu\Sigma$。每个外积都半正定，所以 $W\succeq0$。
> 当 $\Sigma\succ0$ 时，非奇异 Wishart 密度还把自由度延拓到任意实数 $\nu>p-1$；此时非整数 $\nu$ 不再表示向量个数。
> <!-- bilingual-en:start -->
> At integer degrees of freedom, a Wishart matrix is a sum of independent Gaussian outer products. With a positive-definite scale, its nonsingular density extends the family to every real $\nu>p-1$.
> <!-- bilingual-en:end -->

当 $p=1$、$\Sigma=\sigma^2$ 时，
$$
\frac{W}{\sigma^2}=\sum_{i=1}^{\nu}\left(\frac{Z_i}{\sigma}\right)^2\sim\chi^2_\nu.
$$
因此 Wishart 是卡方平方和的矩阵版本：对角元累加各坐标平方，非对角元累加交叉乘积。

$\nu$ 在这个生成式里是独立 Gaussian 向量的个数，因此这里必须是正整数。若 $\Sigma\succ0$ 但整数 $\nu<p$，这个外积和仍有定义，却集中在秩为 $\nu$ 的奇异矩阵上，并没有相对于正定对称矩阵空间的普通密度。常用的非奇异 Wishart 密度可把自由度延拓到任意实数 $\nu>p-1$；这时非整数 $\nu$ 不再表示“有 $\nu$ 个向量”。因此生成式、正定密度与可逆条件必须分别核对。

> [!question]- 自检
> 为什么 Wishart 矩阵必为半正定？
>
> **答案：** 对任意 $a$，$a^TWa=\sum_i(a^TZ_i)^2\ge0$；它是若干秩一半正定外积之和。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|多元统计课程 §1.6.1]]：核对 Gaussian 外积和定义及一元卡方特例。
- [NIST Dataplot, Wishart Random Numbers](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/wishrand.htm)：核对 Wishart 作为多元正态平方和与卡方推广的生成解释。
- [Stanford STATS 305C, *One sample problem*](https://web.stanford.edu/class/stats305c/lectures/Onesample.html#wishart-distribution)：核对独立中心 Gaussian 外积和、$E[W]=\nu\Sigma$ 与样本协方差缩放。
