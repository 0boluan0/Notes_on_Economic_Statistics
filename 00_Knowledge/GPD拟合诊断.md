---
aliases:
  - "GPD拟合诊断分别检查超额分布形状时间依赖与样本外风险预测"
  - GPD model diagnostics
student_os: knowledge-atom
atom_id: RM-EVT-015
atom_type: method
status: source-checked
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
requires:
  - "[[GPD最大似然估计]]"
related:
  - "[[POT阈值选择]]"
  - "[[极端聚集处理]]"
  - "[[Kupiec无条件覆盖]]"
  - "[[Christoffersen独立性]]"
---

# GPD拟合诊断分别检查超额分布形状时间依赖与样本外风险预测
<!-- bilingual-en:start -->
*GPD diagnostics separately examine the excess distribution, temporal dependence, and out-of-sample risk forecasts.*
<!-- bilingual-en:end -->

拟合得到参数后，应先检查所有超额都在支持内，再问模型的概率和分位数是否与观测相容。令 $y_{(1)}\le\cdots\le y_{(k)}$ 是超额样本，$\widehat G$ 是拟合的条件 GPD，取一种明确的作图位置 $p_i=(i-0.5)/k$：
<!-- bilingual-en:start -->
After estimation, first check that all excesses lie in the support, then compare model probabilities and quantiles with observations. Let $y_{(1)}\le\cdots\le y_{(k)}$ be ordered excesses, $\widehat G$ the fitted conditional GPD, and choose an explicit plotting-position convention such as $p_i=(i-0.5)/k$:
<!-- bilingual-en:end -->

$$
\text{Q–Q}:\quad (\widehat G^{-1}(p_i),y_{(i)}),
\qquad
\text{P–P}:\quad (p_i,\widehat G(y_{(i)})).
$$

在此坐标约定下，两种图都与对角线比较。Q–Q 保留超额的幅度单位，能显露大超额的偏离；P–P 使用概率单位，接近 1 的差距可能压缩显著的损失幅度差。它们比较的是条件超额分布，不是直接比较整个损失分布。
<!-- bilingual-en:start -->
Under this convention, both plots are compared with the diagonal. Q–Q retains excess-loss units and exposes magnitude discrepancies; P–P uses probability units, which can compress large loss differences near probability one. These plots concern conditional excesses, not the full loss distribution directly.
<!-- bilingual-en:end -->

还要看超阈值指示变量与超额的时间位置。把同一批超额任意重排，Q–Q 和 P–P 完全不变，连续危机日形成的聚集却可能被打散。因此分布图不能验证 iid；时间结构应另按[[极端聚集处理]]与残差诊断检查。
<!-- bilingual-en:start -->
Also inspect when exceedances and large excesses occur. Arbitrarily permuting the same excesses leaves both plots unchanged while potentially dispersing a crisis cluster. Distribution plots therefore cannot validate iid assumptions; examine temporal structure through [[极端聚集处理|extreme-clustering checks]] and residual diagnostics.
<!-- bilingual-en:end -->

用同一批数据估计参数并画图属于样本内检查。若通过模拟构造拟合诊断的参考带，模拟与重拟合应反映实际估计过程；给定已知参数的参考分布不能直接当作估计参数后的参考分布。用于每日风险预测时，再用时间顺序保留的样本外结果检查覆盖与聚集。[[Kupiec无条件覆盖]]只看例外率，[[Christoffersen独立性]]只针对指定的一阶依赖备择；任一未拒绝都不能单独证明整个 GPD 尾部正确。
<!-- bilingual-en:start -->
Estimating and plotting on the same observations is an in-sample check. Simulation-based diagnostic bands should reproduce the fitting procedure; a reference distribution for known parameters is not automatically valid after estimating them. Daily risk forecasts also need time-ordered out-of-sample checks of coverage and clustering. [[Kupiec无条件覆盖|Kupiec coverage]] examines the exception rate, while [[Christoffersen独立性|Christoffersen independence]] targets a specified first-order dependence alternative. Non-rejection of either test does not establish the full GPD tail model.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Belzile，EVA 2023《Likelihood-based inference》，“Generalized Pareto distribution”](https://lbelzile.github.io/EVA2023-Rtutorial/content/likelihood.html#generalized-pareto-distribution)：核对 GPD 拟合后 P–P/Q–Q 检查。具体作图位置为本卡明确选择；重排不改变两图的结论由次序统计量直接核验。
  <!-- bilingual-en:start -->
  The GPD section supplies post-fit P–P/Q–Q diagnostics. This card explicitly chooses its plotting positions; invariance under permutation follows directly from the order statistics.
  <!-- bilingual-en:end -->
- [Belzile，UNIL 2025，“Metric-based adjustment”](https://lbelzile.github.io/UNIL-2025-choosing-threshold/UNIL-choosing_threshold.html#metric-based-adjustment)：支持通过模拟与重新拟合处理 Q–Q 位置的参数估计不确定性。
  <!-- bilingual-en:start -->
  This section supports simulation with refitting to account for parameter uncertainty in Q–Q positions.
  <!-- bilingual-en:end -->
- [McNeil–Frey，作者版 PDF 第 6、11–13 页](https://statmath.wu.ac.at/~frey/publications/evt-garch.pdf#page=12)：核对过滤残差的时间诊断、滚动一步预测和样本外例外检查；检验职责进一步复用上述覆盖与独立性原子。
  <!-- bilingual-en:start -->
  These pages support residual time-series diagnostics, rolling one-step forecasts, and out-of-sample exception checks. The linked coverage and independence atoms specify the tests' separate responsibilities.
  <!-- bilingual-en:end -->
