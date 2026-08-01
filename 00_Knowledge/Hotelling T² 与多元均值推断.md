---
aliases:
  - "Hotelling T-squared"
  - "Hotelling's T-squared"
  - "多元均值检验"
status: source-checked
---

# Hotelling T² 与多元均值推断
<!-- bilingual-en:start -->
*Hotelling's T-squared and inference for multivariate means*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 同时比较多个相关结果变量的总体均值，而不是分别做许多单变量检验。
> **具体锚点：** 一个教育项目同时影响数学、阅读和写作成绩；三次 t 检验忽略相关性并扩大总体误报风险。
> **核心难点：** 检验对象是均值向量或其线性对比，统计量依赖组内协方差与样本量；显著后仍需解释差异方向。
> **为什么重要：** 它把“总体是否有多维差异”和“哪些变量/对比驱动差异”分成两个层次。
> **继续：** 先掌握 Hotelling $T^2$，再进入 MANOVA；多个连续响应的建模见 [[多元线性回归]]。
> <!-- bilingual-en:start -->
> **What it solves:** It compares a population mean vector, or two mean vectors, while accounting for correlation among outcomes.
> **Concrete anchor:** An educational programme may affect mathematics, reading, and writing scores jointly. Three separate t-tests ignore their correlation and inflate the chance of at least one false positive.
> **Central difficulty:** The target is an entire mean vector or a prespecified linear contrast. The statistic depends on covariance and sample size, and a significant omnibus result still requires localisation.
> **Why it matters:** It separates the global question of whether a multidimensional difference exists from the follow-up question of which outcomes or contrasts create it.
> **Continue with:** Use [[MANOVA 多元方差分析|MANOVA]] when more than two groups or factorial effects are the target, and [[多元线性回归|multivariate linear regression]] for general predictor matrices.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [Penn State STAT 505](https://online.stat.psu.edu/stat505/)：核验多元正态、均值推断、MANOVA、PCA、因子分析、判别与聚类的定义和使用条件。
> - Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验矩阵公式与抽样分布。
> <!-- bilingual-en:start -->
> - [Penn State STAT 505](https://online.stat.psu.edu/stat505/) was checked for the definitions and assumptions of multivariate mean inference.
> - Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was checked for matrix formulas and exact sampling distributions.
> <!-- bilingual-en:end -->

## 单总体 Hotelling $T^2$
<!-- bilingual-en:start -->
*One-sample Hotelling's $T^2$*
<!-- bilingual-en:end -->

检验 $H_0:\mu=\mu_0$ 时，$T^2=n(\bar x-\mu_0)^TS^{-1}(\bar x-\mu_0)$ 用协方差校正后的距离衡量偏离。正态且 $n>p$ 等条件下可转为 F 分布。它是多元 t 检验，但不能在 S 奇异时直接使用。
<!-- bilingual-en:start -->
For $H_0:\mu=\mu_0$, $T^2=n(\bar x-\mu_0)^TS^{-1}(\bar x-\mu_0)$ measures a covariance-adjusted distance. Under multivariate normality and conditions such as $n>p$, it transforms to an $F$ distribution. It is the multivariate analogue of a t-test, but it cannot be used directly when $S$ is singular.
<!-- bilingual-en:end -->

在单样本情形，
$$
\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}
$$
。这个转换揭示了维度成本：$p$ 越接近 $n$，协方差求逆越不稳，分母自由度 $n-p$ 也越小。“多收集一个响应”并非免费，必须确认它真的回答同一科学问题。
<!-- bilingual-en:start -->
In the one-sample case,
$$
\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}.
$$
This transformation exposes the cost of dimensionality. As $p$ approaches $n$, covariance inversion becomes unstable and the denominator degrees of freedom $n-p$ shrink. Adding another outcome is not free; it should answer the same scientific question.
<!-- bilingual-en:end -->

## 配对与两独立总体
<!-- bilingual-en:start -->
*Paired and two-independent-sample designs*
<!-- bilingual-en:end -->

配对设计先对每对观察取差向量，再做单总体 $T^2$；独立组比较则基于均值差。经典 pooled 版本假设协方差矩阵相同，若不可信应采用稳健/近似方法并说明口径。
<!-- bilingual-en:start -->
For paired observations, form a difference vector for each pair and then apply the one-sample $T^2$ test. For independent groups, work with the difference between sample mean vectors. The classical pooled version assumes equal covariance matrices; if that assumption is implausible, use an appropriate robust or approximate method and state the inferential convention.
<!-- bilingual-en:end -->

两独立组且协方差相同时，使用 pooled 协方差 $S_p$：
$$
T^2=\frac{n_1n_2}{n_1+n_2}(\bar x_1-\bar x_2)^TS_p^{-1}(\bar x_1-\bar x_2).
$$
系数 $n_1n_2/(n_1+n_2)$ 是均值差方差的样本量校正；若两组样本量都很小，协方差差异会同时扰动距离和参考分布。
<!-- bilingual-en:start -->
For two independent groups with a common covariance matrix, use the pooled covariance $S_p$:
$$
T^2=\frac{n_1n_2}{n_1+n_2}(\bar x_1-\bar x_2)^TS_p^{-1}(\bar x_1-\bar x_2).
$$
The factor $n_1n_2/(n_1+n_2)$ adjusts for the variance of the difference in sample means. With two small samples, unequal covariances disturb both the distance and its reference distribution.
<!-- bilingual-en:end -->

## 置信区域与同时区间
<!-- bilingual-en:start -->
*Confidence regions and simultaneous intervals*
<!-- bilingual-en:end -->

$T^2$ 给均值向量的椭球置信区域。若关心各分量或线性组合，可构造同时区间；Bonferroni 用较保守的单项水平控制家族错误率。先定义实际关心的对比，避免显著后无限搜索。
<!-- bilingual-en:start -->
$T^2$ yields an ellipsoidal confidence region for a mean vector. If individual components or linear combinations matter, simultaneous intervals can be constructed; Bonferroni uses more conservative componentwise levels to control family-wise error. Define scientifically relevant contrasts in advance rather than searching without limit after significance.
<!-- bilingual-en:end -->

椭球回答“整个向量在哪里”，区间回答“某个坐标或方向在哪里”。两者不是重复输出：前者适合全局不确定性，后者适合可解释的实质结论。
<!-- bilingual-en:start -->
The ellipsoid answers where the whole vector may lie; an interval answers where one coordinate or direction may lie. These are not redundant outputs: the first represents global uncertainty, whereas the second supports interpretable substantive conclusions.
<!-- bilingual-en:end -->

## 工作流与诊断
<!-- bilingual-en:start -->
*Workflow and diagnostics*
<!-- bilingual-en:end -->

1. 由研究问题定义响应向量与对比，不要为提高显著性事后换变量。
2. 检查配对/独立结构、缺失、多元异常点、样本量与 $S$ 的秩。
3. 根据设计选择单样本、配对或两样本统计量，并说明正态与等协方差假设。
4. 报告全局检验、效应方向、同时区间和敏感性分析，而不只报 $p$ 值。
<!-- bilingual-en:start -->

&nbsp;
**1.** Define the response vector and contrasts from the research question; do not swap outcomes after seeing significance.<br>
**2.** Check pairing or independence, missingness, multivariate outliers, sample size, and the rank of $S$.<br>
**3.** Select a one-sample, paired, or two-sample statistic from the design and state normality and equal-covariance assumptions.<br>
**4.** Report the global test, effect directions, simultaneous intervals, and sensitivity analyses rather than only a p-value.<br>
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 为什么分别做 p 个 t 检验不等同于一次多元检验？
<!-- bilingual-en:start -->
*Why are $p$ separate t-tests not equivalent to one multivariate test?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它忽略变量相关结构且若不校正会膨胀家族错误率；多元检验针对整个均值向量。
> <!-- bilingual-en:start -->
> They ignore dependence among outcomes and, without adjustment, inflate family-wise error. A multivariate test targets the mean vector as a whole.
> <!-- bilingual-en:end -->

### 配对多元数据为什么先取差向量？
<!-- bilingual-en:start -->
*Why are difference vectors formed first for paired multivariate data?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 配对依赖由同一对象内的差消化，问题转成差向量总体均值是否为零。
> <!-- bilingual-en:start -->
> Within-unit differencing absorbs the paired dependence and turns the question into whether the population mean of the difference vector is zero.
> <!-- bilingual-en:end -->

### 为什么 $p\ge n$ 时不能直接做经典 Hotelling $T^2$？
<!-- bilingual-en:start -->
*Why can the classical Hotelling $T^2$ not be applied directly when $p\ge n$?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 中心化样本协方差的秩至多为 $n-1$，因而 $S$ 奇异、$S^{-1}$ 不存在，参考 F 分布的自由度也失效。
> <!-- bilingual-en:start -->
> A centred sample covariance matrix has rank at most $n-1$, so $S$ is singular, $S^{-1}$ does not exist, and the reference $F$ distribution no longer has valid degrees of freedom.
> <!-- bilingual-en:end -->

### 用自己的话说明“全局显著”为什么不等于“每个响应都显著”。
<!-- bilingual-en:start -->
*Explain in your own words why global significance does not mean that every outcome is significant.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 全局检验可由某一个响应或几个响应的线性组合驱动；它只否定整个均值向量假设，并没有分别否定每个分量假设。
> <!-- bilingual-en:start -->
> The global result may be driven by one outcome or by a linear combination of several outcomes. It rejects a vector hypothesis, not every componentwise hypothesis separately.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Penn State STAT 505](https://online.stat.psu.edu/stat505/)：核验多元正态、均值推断、MANOVA、PCA、因子分析、判别与聚类的定义和使用条件。
- Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验矩阵公式与抽样分布。
<!-- bilingual-en:start -->
- [Penn State STAT 505](https://online.stat.psu.edu/stat505/) was checked for the definition, designs, and assumptions of multivariate mean inference.
- Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was checked for matrix formulas and exact sampling distributions.
<!-- bilingual-en:end -->

- [Penn State STAT 505, Lesson 7](https://online.stat.psu.edu/stat505/Lesson07)：逐项核验单样本、配对与两样本 Hotelling $T^2$、F 转换、置信椭球和同时区间。
<!-- bilingual-en:start -->
- [Penn State STAT 505, Lesson 7](https://online.stat.psu.edu/stat505/Lesson07) was checked item by item for one-sample, paired, and two-sample Hotelling $T^2$, its $F$ transformation, confidence ellipsoids, and simultaneous intervals.
<!-- bilingual-en:end -->
