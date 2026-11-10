---
aliases:
  - "Multivariate Gaussian diagnostic workflow"
  - "多元正态性诊断"
  - "Joint Gaussian diagnostics"
student_os: knowledge-atom
atom_id: STAT-MVN-007
atom_set: multivariate-normal
atom_type: diagnostic-workflow
status: source-checked
requires:
  - "[[联合Gaussian]]"
  - "[[Gaussian马氏平方律]]"
related:
  - "[[边际与马氏图不证联合Gaussian]]"
  - "[[多元数据诊断]]"
  - "[[马氏距离]]"
  - "[[异常观测处理原则]]"
part_of:
  - "[[多元正态分布.canvas]]"
---

<!-- bilingual-en:start -->
# Assessing a joint Gaussian model

A Gaussian diagnostic asks whether a proposed probability model describes the features of the data that matter for an analysis. It looks for particular disagreements—such as a curved relationship or unusually long tails—and traces what those disagreements would change. It does not turn a collection of attractive plots into a proof of the model.

The model concerns a [[随机向量|random vector]] $X=(X_1,\ldots,X_p)^T$: one observation produces $p$ measurements together. [[联合Gaussian|Joint Gaussianity]] requires every fixed real linear combination of these measurements to be normal, allowing constants as degenerate normals. Checking each measurement separately therefore covers only part of the assumption.
<!-- bilingual-en:end -->

*联合 Gaussian 诊断*

Gaussian 诊断是在问：一个拟议的概率模型，能否描述当前分析关心的数据特征？它寻找具体的不相容之处，例如弯曲的关系或特别长的尾部，并追查这些偏离会改变什么结论。几张好看的图并不能构成模型成立的证明。

模型针对[[随机向量]] $X=(X_1,\ldots,X_p)^T$：一次观测同时产生 $p$ 个测量量。[[联合Gaussian|联合 Gaussian]] 要求这些测量量的每个固定实线性组合都正态，常数则视为退化正态。因此，分别检查每个变量，只覆盖了假设的一部分。

<!-- bilingual-en:start -->
## A sequence with a reason for each step
<!-- bilingual-en:end -->

## 一套最小但完整的顺序

<!-- bilingual-en:start -->
Start by stating the object and purpose. For example, suppose two instruments measure each of 80 manufactured items. A row records the paired readings $(x_{i1},x_{i2})$ for item $i$, so $n=80$ is the number of items and $p=2$ the number of variables. If the intended calculation is the probability of an unusually large average reading, the distribution of $(X_1+X_2)/2$ and its upper tail matter directly.

Before plotting, check item identifiers, units, missing-value codes, measurement limits and production batches. Two readings from one item belong together; 80 repeated readings of one item do not become 80 independent items. The [[数据矩阵不推出抽样模型|table layout does not establish the sampling model]]. A Gaussian law within a row and independence between rows are separate assumptions. Time order, clusters and repeated measurements must remain visible. The broader [[多元数据诊断|data diagnostic procedure]] explains these checks before any particular distribution is chosen.
<!-- bilingual-en:end -->

先说明对象和用途。例如，用两台仪器分别测量 80 件产品，每行记录第 $i$ 件产品的成对读数 $(x_{i1},x_{i2})$；此时 $n=80$ 是产品数，$p=2$ 是变量数。若目标是计算“平均读数异常偏大”的概率，那么 $(X_1+X_2)/2$ 的分布及其右尾就直接相关。

画图之前，先检查产品编号、单位、缺失编码、测量上限及生产批次。同一产品的两个读数要保留配对；对同一件产品重复读 80 次，并不会变成 80 件独立产品。[[数据矩阵不推出抽样模型|表格形状不能建立抽样模型]]，一行内部服从 Gaussian 与各行相互独立是两项不同假设。时间顺序、群组和重复测量结构都应保留；模型选择之前的通用检查见[[多元数据诊断]]。

<!-- bilingual-en:start -->
Next inspect each coordinate. A histogram or box plot can reveal skewness, multiple modes, boundary piles or extreme values. A normal probability plot, often called a normal Q–Q plot, orders that coordinate's observations and compares them with theoretical [[正态分布|normal]] [[总体分位数|quantiles]]—values corresponding to specified cumulative probabilities. A roughly straight pattern is compatible with a normal shape; the line need not have slope one because location and scale may be unknown. Systematic curvature or tail departures identify features to investigate. They do not identify a unique cause.

Then examine the pairing. Plot one instrument against the other, preserve batch labels, and inspect useful combinations such as the average and the difference $X_1-X_2$. Two bell-shaped margins can hide an unusual dependence pattern. For a nondegenerate bivariate Gaussian model the population density contours are ellipses; a curved band, separated groups or a fan-shaped spread gives a reason to reconsider a single such model. In higher dimensions, pairwise plots and a few selected projections remain partial views. Even the direction chosen by looking at the data is an exploratory choice, not an automatically calibrated new test.
<!-- bilingual-en:end -->

接着分别看每个坐标。直方图或箱线图能提示偏斜、多峰、边界堆积或极端值。正态概率图通常也称正态 Q–Q 图，它把该变量的观测排序后，与理论[[正态分布]]的[[总体分位数|分位数]]比较；分位数就是与指定累积概率对应的位置。大致成直线说明形状与正态相容；位置与尺度未知时，直线不必斜率为一。系统性弯曲或尾部偏离提示需要调查的特征，却不能唯一确定原因。

随后检查配对关系。把两台仪器的读数画成散点图，保留批次标签，再查看平均值以及差值 $X_1-X_2$ 等有实际意义的组合。两个钟形边际可能隐藏异常依赖。非退化二维 Gaussian 的总体密度等高线是椭圆；弯曲带、分开的群组或扇形散布，都提示需要重新考虑用一个这样的分布描述全部数据。在更高维下，成对图和少量投影仍只是局部视角。看完数据才挑出的方向属于探索，不能自动当作已经校准的新检验。

<!-- bilingual-en:start -->
When a full covariance estimate is usable, add a distance view. For $n\ge2$ complete real observations with $p$ coordinates, let $\bar x$ be their [[样本均值向量|sample mean]] and $S$ their [[样本协方差矩阵|sample covariance]] with denominator $n-1$. If $S$ is invertible, calculate the squared [[马氏距离|Mahalanobis distances]]
$$d_i^2=(x_i-\bar x)^TS^{-1}(x_i-\bar x).$$
These measure displacement after accounting for scale and correlation. Order them as $d_{(1)}^2\le\cdots\le d_{(n)}^2$ and compare them with quantiles of the [[卡方分布|chi-square distribution]] with $p$ degrees of freedom. One common plotting position is $u_i=(i-0.5)/n$; use the chi-square quantile at probability $u_i$ on the horizontal axis. This avoids the infinite upper endpoint at probability one.

The [[Gaussian马氏平方律|exact Gaussian chi-square law]] uses the true fixed population mean and covariance. Here both are fitted to the same observations, so the chi-square plot is an approximate shape reference, not a set of independent exact chi-square draws or an automatic outlier threshold. An upward bend at the upper right suggests larger fitted distances than the reference, worth examining alongside the original records and other plots.
<!-- bilingual-en:end -->

协方差可以有效估计时，再加入距离视角。对 $n\ge2$ 个完整的 $p$ 维实数观测，以 $\bar x$ 表示[[样本均值向量|样本均值]]，以 $S$ 表示分母为 $n-1$ 的[[样本协方差矩阵|样本协方差]]。若 $S$ 可逆，计算平方[[马氏距离]]
$$d_i^2=(x_i-\bar x)^TS^{-1}(x_i-\bar x).$$
它考虑尺度和相关性后衡量偏离中心的程度。把距离平方排成 $d_{(1)}^2\le\cdots\le d_{(n)}^2$，与自由度为 $p$ 的[[卡方分布]]分位数比较。一种常用绘图位置是 $u_i=(i-0.5)/n$，横轴使用累积概率 $u_i$ 对应的卡方分位数；这样避开了概率为一时的无穷大上端点。

[[Gaussian马氏平方律|Gaussian 的精确卡方律]]使用真实、固定的总体均值与协方差。这里却用同一批观测估计了两者，因此卡方图只是近似形状参照，并非一组相互独立、精确服从卡方的抽样值，也不自动给出异常点阈值。右上方向上弯曲，提示拟合距离比参照更大，需要结合原始记录和其他图调查。

> [!example] More variables than observations · 变量多于观测
> <!-- bilingual-en:start -->
> With $n=20$ observations and $p=30$ variables, the [[样本协方差秩边界|sample covariance rank]] is at most $n-1=19$, so the ordinary inverse does not exist. Marginal, low-dimensional projection and provenance checks can still proceed. Replacing the inverse by a pseudoinverse or regularized inverse changes the distance and requires its own reference law. Even when $p<n$, invertibility alone does not guarantee a stable estimate.
> <!-- bilingual-en:end -->
>
> 若 $n=20$、$p=30$，[[样本协方差秩边界|样本协方差的秩]]至多为 $n-1=19$，所以普通逆不存在。但仍能检查边际、低维投影和数据来源。换成伪逆或正则化逆会改变距离，需要另外建立参照分布；即使 $p<n$，仅仅可逆也不保证估计稳定。

<!-- bilingual-en:start -->
Finally, return to the intended calculation. In the instrument example, compare how plausible modeling choices change the estimated upper-tail probability of the average, rather than judging only which plot looks straighter. A justified transformation, a model for distinct batches, an alternative tail model or a suitable robust procedure answers different questions and brings additional assumptions. Record those choices and compare the same scientific target where possible; transforming the response may change the target itself.
<!-- bilingual-en:end -->

最后回到实际计算。在仪器例子中，应比较合理模型选择怎样改变“平均读数过大”的估计概率，而不是只比较哪张图更直。合理变换、分别建模不同批次、替代尾部分布或适当的稳健方法，回答的问题可能不同，也各自带来附加假设。应记录选择，并尽可能比较同一个实际目标；变换响应变量本身可能改变目标。

<!-- bilingual-en:start -->
## Interpreting and reporting what you found
<!-- bilingual-en:end -->

## 怎样解释结果

<!-- bilingual-en:start -->
A distant point may be a recording error, an item outside the intended population, a genuine rare event, a separate subgroup or a missing feature of the model. Follow the [[异常观测处理原则|flagged-observation procedure]]: investigate first, then justify correction, exclusion, a model change or retention with a sensitivity analysis. Removing observations merely to improve a Gaussian plot hides the feature being diagnosed.

A formal normality test adds evidence only under its own sampling and calibration assumptions. Small samples can miss substantial departures; large samples can detect deviations with little effect on the intended calculation. A useful report states the features found, the directions and tails examined, any dimensional or sampling limitations, and the effect on the analysis. For example: “The coordinate plots were approximately straight, but the difference showed two batch-related groups; a single joint Gaussian model was therefore reconsidered.” The separate [[边际与马氏图不证联合Gaussian|diagnostic boundary]] explains why even reassuring checks do not establish the population law.
<!-- bilingual-en:end -->

一个距离很远的点，可能是记录错误、目标总体之外的对象、真实稀有事件、另一个子群，或模型漏掉了某项结构。应按[[异常观测处理原则]]先调查，再说明为什么更正、排除、改变模型，或保留并报告敏感性。仅为了让 Gaussian 图更好看而删点，会隐藏正在诊断的特征。

形式正态性检验也只有在自身抽样与校准条件下才提供证据。小样本可能漏掉重要偏离，大样本可能检出几乎不影响实际计算的偏离。有用的报告应说明发现了什么、看过哪些方向与尾部、维度或抽样有什么限制，以及这些问题怎样影响分析。例如：“各坐标图大致成直线，但差值呈现两个与批次相关的群组，因此重新考虑了单一联合 Gaussian 模型。”即使结果看起来令人放心，也不能确立总体分布，原因见[[边际与马氏图不证联合Gaussian|诊断边界]]。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.8. 正态性假设检验|Multivariate statistics course, §1.8]]: course scope for marginal and Mahalanobis plots and fitted-distance cautions.
- [Penn State STAT 505, Lesson 3, §3.1](https://online.stat.psu.edu/stat505/Lesson03): coordinate/pairwise graphics, transformations and investigating unusual observations; [Lesson 4, §4.4](https://online.stat.psu.edu/stat505/Lesson04): ordered squared-distance plots, plotting positions and the approximate chi-square reference for fitted distances.
- [NIST/SEMATECH, Normal Probability Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm): ordered observations versus theoretical normal quantiles and interpretation of curvature and tails. [Quantile–Quantile Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/eda33o.htm) distinguishes a two-sample Q–Q plot from a theoretical probability plot.
- [Penn State STAT 200, Lesson 6, §§6.1 and 6.5](https://online.stat.psu.edu/stat200/Lesson06): errors, non-rejection and power as a function of sample size and effect size. [[Gaussian马氏平方律|The Gaussian distance law]] and [[样本协方差秩边界|sample covariance rank]] supply the exact mathematical boundaries; the instrument scenario is illustrative, not observed data.
<!-- bilingual-en:end -->

- 多元统计课程 §1.8 确定边际图、马氏图及拟合距离限制的课程范围。
- Penn State STAT 505 第3课支持坐标与成对图形、变换和异常来源调查；第4课 §4.4 支持排序平方距离、绘图概率及拟合距离的近似卡方参照。
- NIST 正态概率图条目支持理论正态分位数与观测排序比较、弯曲及尾部的解释；Q–Q 条目区分两样本分位图与理论概率图。
- Penn State STAT 200 第6课支持不拒绝、错误与功效边界；精确距离律和秩条件由对应卡片及其已核原源支持。仪器情境是讲解例子，不是实际数据分析结果。
