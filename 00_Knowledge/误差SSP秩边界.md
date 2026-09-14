---
aliases:
  - "多响应误差 SSP 的秩至多为 min(q,n-rank(X))"
  - The error SSP rank is at most min(q,n-rank(X))
  - high-dimensional response rank boundary
student_os: knowledge-atom
atom_id: STAT-MVR-008
atom_set: multivariate-linear-regression
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[多元线性回归.canvas|多元线性回归]]"
requires:
  - "[[多响应残差与误差SSP]]"
  - "[[矩阵秩]]"
  - "[[秩零度定理]]"
related:
  - "[[Wishart秩与可逆性]]"
  - "[[PCA低秩重构]]"
---

# 多响应误差 SSP 的秩至多为 min(q,n-rank(X))
<!-- bilingual-en:start -->
*The multivariate error SSP has rank at most $\min(q,n-\operatorname{rank}(X))$*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 令 $r=\operatorname{rank}(X)$、$R=(I-P_X)Y$。由于残差只落在观测空间中维数 $n-r$ 的正交补，
> $$\operatorname{rank}(E_{SSP})=\operatorname{rank}(R^TR)=\operatorname{rank}(R)
> \le \min(q,n-r).$$
> 因而 $q>n-r$ 时，$E_{SSP}$ 必奇异，不能直接使用需要 $E_{SSP}^{-1}$ 的经典全响应检验。
> <!-- bilingual-en:start -->
> Residualisation leaves only $n-r$ observational directions, so the $q\times q$ error SSP cannot have rank above $\min(q,n-r)$.
> <!-- bilingual-en:end -->

在独立 Gaussian 行、$\Sigma\succ0$ 下，$E_{SSP}\sim W_q(\Sigma,n-r)$，并几乎必然达到这个上界。$q\le n-r$ 只使满秩在维度上成为可能；在上述连续 Gaussian 条件下才进一步得到几乎处处可逆，任意实际数据仍可能因精确共线而秩亏。即便可逆，当 $q$ 接近 $n-r$ 时，最小特征值和逆矩阵仍可能很不稳定；“能求逆”不是“协方差估得可靠”。

高维小样本不能靠把奇异矩阵硬求逆解决。可行方向包括：在看数据前聚焦低维响应对比、使用有明确假设的降维、正则化协方差或采用适配高维的检验。每种方法改变了检验对象或校准条件，不能继续冒充原来的经典全维 MANOVA。

> [!question]- 自检
> $n=30$、$\operatorname{rank}(X)=8$、$q=25$ 时，误差 SSP 最大秩是多少？
>
> **答案：** $\min(25,30-8)=22$，所以 $25\times25$ 的误差 SSP 必奇异。

## 来源与核验

- [[多响应残差与误差SSP]] 与 [[Wishart秩与可逆性]]：核对残差自由度 $n-r$、Wishart 秩和可逆门槛。
- [R `summary.manova` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.manova.html)：核对实现会检查 residual correlation matrix 的秩亏，高度相关响应会使结果不准确。
- [[矩阵秩]]与[[秩零度定理]]：结合 $\ker(R^TR)=\ker(R)$ 核对二者同秩，并核对投影残差空间的维数 $n-r$。
<!-- bilingual-en:start -->
- The residual-SSP and Wishart atoms give the algebraic and probabilistic rank bounds; R's official documentation confirms that residual rank deficiency is a practical validity boundary.
<!-- bilingual-en:end -->
