---
aliases:
  - "单因素 MANOVA 把总 SSCP 分解为组间假设 SSCP 与组内误差 SSCP"
  - "MANOVA SSCP decomposition"
  - "MANOVA 的 H、E 与 T 矩阵"
student_os: knowledge-atom
atom_id: STAT-MAN-002
atom_set: manova
atom_type: decomposition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
requires:
  - "[[MANOVA]]"
  - "[[多响应残差与误差SSP]]"
implies:
  - "[[MANOVA四种统计量]]"
related:
  - "[[误差SSP秩边界]]"
  - "[[Wishart分布]]"
---

# 单因素 MANOVA 把总 SSCP 分解为组间假设 SSCP 与组内误差 SSCP
<!-- bilingual-en:start -->
*One-way MANOVA decomposes total SSCP into between-group hypothesis SSCP and within-group error SSCP*
<!-- bilingual-en:end -->

> [!summary] 原子分解
> 令第 $i$ 组有 $n_i$ 个 $p$ 维观测，$N=\sum_i n_i$，组均值为 $\bar Y_{i\cdot}$，加权总均值为 $\bar Y_{\cdot\cdot}=N^{-1}\sum_i n_i\bar Y_{i\cdot}$。定义
> $$
> E=\sum_{i=1}^g\sum_{j=1}^{n_i}(Y_{ij}-\bar Y_{i\cdot})(Y_{ij}-\bar Y_{i\cdot})^T,
> $$
> $$
> H=\sum_{i=1}^g n_i(\bar Y_{i\cdot}-\bar Y_{\cdot\cdot})(\bar Y_{i\cdot}-\bar Y_{\cdot\cdot})^T.
> $$
> 则总 SSCP 满足
> $$T=E+H,$$
> 自由度分别为 $N-g$、$g-1$ 与 $N-1$。
> <!-- bilingual-en:start -->
> Within-group error SSCP $E$ and between-group hypothesis SSCP $H$ add to total SSCP $T$, with degrees of freedom $N-g$, $g-1$, and $N-1$.
> <!-- bilingual-en:end -->

$E$ 把每个观测与**本组均值**的偏差外积相加，描述完整组别模型仍未解释的响应共变；$H$ 把每个**组均值与总均值**的偏差外积按组样本量加权，描述组别效应对应的响应共变。二者都是 $p\times p$ 对称半正定矩阵。这里的 $E$ 是压缩后的误差 SSCP，不是逐观测残差矩阵，见 [[多响应残差与误差SSP]]。现有卡名沿用较短的 SSP；两者在本库都指 sums of squares and cross-products。

分解来自恒等式
$$Y_{ij}-\bar Y_{\cdot\cdot}=(Y_{ij}-\bar Y_{i\cdot})+(\bar Y_{i\cdot}-\bar Y_{\cdot\cdot}).$$
展开外积后，交叉项因为每组内 $\sum_j(Y_{ij}-\bar Y_{i\cdot})=0$ 而消失。它是多元版“总平方和 = 组内平方和 + 组间平方和”，但非对角元还保留不同响应残差的交叉乘积，因而这里采用更明确的 SSCP 缩写。

秩先于求逆：
$$\operatorname{rank}(H)\le \min(p,g-1),\qquad
\operatorname{rank}(E)\le \min(p,N-g).$$
因此 $N-g<p$ 时 $E$ 必奇异；即使维数门槛满足，实际响应精确共线也会使 $E$ 秩亏。后续统计量不能把“有一个 $p\times p$ 矩阵”误当成“它一定可逆”，见 [[误差SSP秩边界]]。

> [!question]- 自检
> 四组、总样本量 $N=40$ 时，$H$、$E$、$T$ 的自由度各是多少？
>
> **答案：** 分别是 $3$、$36$、$39$，并且 $T=H+E$。

## 来源与核验

- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.5.3. SSCP 矩阵分解|本地多元统计课程 §1.5.3–1.5.5]]：核对 $T=H+E$、MANOVA 表和自由度。
- [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08)：核对 one-way MANOVA 的 treatment、error、total SSCP 及 $g-1,N-g,N-1$ 自由度。
- [SAS GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm)：核对一般线性假设下 $H$ 与 $E$ 的构造，以及对角元对应逐响应 hypothesis/error sums of squares。
- [[误差SSP秩边界]]：核对误差 SSCP 的残差自由度和可逆性门槛。
<!-- bilingual-en:start -->
- The local course and Penn State verify the one-way SSCP table; the SAS GLM documentation verifies the general hypothesis/error SSCP interpretation.
<!-- bilingual-en:end -->
