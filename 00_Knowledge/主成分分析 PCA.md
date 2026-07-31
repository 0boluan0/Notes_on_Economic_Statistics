---
aliases:
  - "Principal Component Analysis"
  - "PCA"
  - "主成分分析"
status: source-checked
---

# 主成分分析 PCA
<!-- bilingual-en:start -->
*Principal component analysis (PCA)*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** PCA 用少数线性组合压缩观测变异；因子分析用少数潜在因子解释变量间共同相关。
> **具体锚点：** 多个考试科目高度相关，PCA 可总结为几个综合分数；因子分析则会问是否存在“语言能力”“数量能力”等潜在来源。
> **核心难点：** 二者都用载荷却回答不同问题；主成分是数据的确定线性组合，因子是带特殊误差的统计模型。
> **为什么重要：** 错把降维当潜变量解释会产生过度命名和错误因果叙事。
> **继续：** 先明确目标是压缩还是潜在结构，再选择协方差/相关矩阵、成分/因子数和验证方式。
> <!-- bilingual-en:start -->
> **What it solves:** PCA compresses observed variation into a small number of linear combinations, whereas factor analysis explains shared correlations through a small number of latent factors.
> **Concrete anchor:** When several examination subjects are strongly correlated, PCA can summarise them into a few composite scores. Factor analysis instead asks whether latent sources such as verbal and quantitative ability generate their common variation.
> **Central difficulty:** Both methods use loadings but answer different questions. A principal component is a deterministic linear combination of observed data; a factor belongs to a statistical model with specific errors.
> **Why it matters:** Treating dimension reduction as latent-variable discovery leads to overconfident labels and false causal narratives.
> **Continue with:** First decide whether the goal is compression or latent structure, then select a covariance or correlation matrix, a number of components or factors, and a validation method. For the latent-model route, see [[因子分析|factor analysis]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [Penn State STAT 505](https://online.stat.psu.edu/stat505/)：核验多元正态、均值推断、MANOVA、PCA、因子分析、判别与聚类的定义和使用条件。
> - Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验矩阵公式与抽样分布。
> <!-- bilingual-en:start -->
> - [Penn State STAT 505](https://online.stat.psu.edu/stat505/) was used to verify definitions and conditions for multivariate normal models, mean inference, MANOVA, PCA, factor analysis, discriminant analysis, and clustering.
> - Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was used to verify matrix formulas and sampling distributions.
> <!-- bilingual-en:end -->

## PCA 的方差最大化
<!-- bilingual-en:start -->
*PCA as variance maximisation*
<!-- bilingual-en:end -->

第一主成分 $z_1=a_1^TX$ 在 $\lVert a_1\rVert=1$ 下最大化方差，其方向是协方差矩阵最大特征值对应特征向量；后续成分与此前方向正交并依次最大化剩余方差。成分得分是观测的线性组合，解释方差比为 $\lambda_j/\sum_k\lambda_k$。
<!-- bilingual-en:start -->
The first principal component $z_1=a_1^TX$ maximises variance subject to $\lVert a_1\rVert=1$, so its direction is the eigenvector of the covariance matrix associated with the largest eigenvalue. Later components are orthogonal to preceding directions and successively maximise remaining variance. Component scores are linear combinations of observations, and the explained-variance ratio is $\lambda_j/\sum_k\lambda_k$.
<!-- bilingual-en:end -->

这一优化可由 Lagrange 乘子推导：最大化 $a^T\Sigma a$ 且 $a^Ta=1$，一阶条件给 $\Sigma a=\lambda a$。所以 PCA 不是任意旋转，而是协方差结构强制给出的正交方向；特征值就是该方向上的方差。
<!-- bilingual-en:start -->
The optimisation follows from a Lagrange-multiplier argument. Maximising $a^T\Sigma a$ subject to $a^Ta=1$ gives the first-order condition $\Sigma a=\lambda a$. PCA is therefore not an arbitrary rotation: covariance structure determines the orthogonal directions, and each eigenvalue is the variance along its direction.
<!-- bilingual-en:end -->

## 标准化与成分选择
<!-- bilingual-en:start -->
*Standardisation and choosing the number of components*
<!-- bilingual-en:end -->

变量单位相差大时，基于协方差的 PCA 会被大尺度变量主导；相关矩阵 PCA 等价于先标准化。成分数可参考累计解释率、scree plot、平行分析和下游任务，但没有单一机械阈值。
<!-- bilingual-en:start -->
When variables have very different units, covariance-based PCA is dominated by large-scale variables. Correlation-based PCA is equivalent to standardising first. The number of components can be informed by cumulative explained variance, a scree plot, parallel analysis, and downstream performance, but no single mechanical threshold is universally valid.
<!-- bilingual-en:end -->

选择协方差还是相关矩阵不是数据清洗细节，而是改变问题。若原始尺度本身有意义，标准化会给小方差变量与大方差变量同等机会；若尺度是任意单位，标准化通常更合理。必须在看到结果前写清理由。
<!-- bilingual-en:start -->
Choosing covariance rather than correlation is not a minor preprocessing detail; it changes the question. When original scale is substantively meaningful, standardisation gives low-variance and high-variance variables equal opportunity. When scale is largely an artefact of units, standardisation is usually more defensible. The choice should be justified before inspecting preferred results.
<!-- bilingual-en:end -->

## Worked example：两个高度相关变量
<!-- bilingual-en:start -->
*Worked example: two highly correlated variables*
<!-- bilingual-en:end -->

若标准化后的两个变量相关矩阵为 $R=\begin{pmatrix}1&0.8\\0.8&1\end{pmatrix}$，特征值为 1.8 与 0.2，对应方向分别与 $(1,1)$、$(1,-1)$ 成比例。第一成分近似“两变量共同水平”，解释 $1.8/2=90\%$ 的总标准化方差；第二成分是二者差异。
<!-- bilingual-en:start -->
For two standardised variables with correlation matrix $R=\begin{pmatrix}1&0.8\\0.8&1\end{pmatrix}$, the eigenvalues are 1.8 and 0.2, with directions proportional to $(1,1)$ and $(1,-1)$. The first component is approximately their common level and explains $1.8/2=90\%$ of total standardised variance; the second captures their contrast.
<!-- bilingual-en:end -->

载荷符号整体反转不会改变成分；$(1,1)$ 与 $(-1,-1)$ 表示同一轴。解释时应看变量与成分的相关或标准化载荷，并说明方向约定，不能把符号任意性当作结果不稳定。
<!-- bilingual-en:start -->
Flipping all loading signs does not change a component: $(1,1)$ and $(-1,-1)$ describe the same axis. Interpretation should use correlations or standardised loadings and state the sign convention rather than treating sign indeterminacy as instability.
<!-- bilingual-en:end -->

## PCA 与因子分析的选择
<!-- bilingual-en:start -->
*Choosing between PCA and factor analysis*
<!-- bilingual-en:end -->

若目标是压缩、可视化或预测预处理，PCA 往往直接；若有潜变量理论并关心测量误差与共同结构，因子模型更合适。两者都需对样本稳定性、异常值和外部效度负责。
<!-- bilingual-en:start -->
PCA is often the direct choice for compression, visualisation, or predictive preprocessing. When a latent-variable theory motivates shared structure and measurement error matters, a factor model is more appropriate. Both methods require checks of sample stability, outliers, and external validity.
<!-- bilingual-en:end -->

## 解释边界
<!-- bilingual-en:start -->
*Limits of interpretation*
<!-- bilingual-en:end -->

载荷大表示统计关联，不自动证明潜在因子的真实存在或因果方向。成分/因子命名应由变量内容、理论与新样本验证共同支持。
<!-- bilingual-en:start -->
A large loading indicates statistical association; it does not prove that a latent factor exists or establish a causal direction. Labels for components or factors should be supported jointly by variable content, theory, and validation in new samples.
<!-- bilingual-en:end -->

PCA 对异常值和重尾敏感，因为协方差矩阵本身敏感。训练集上估计的中心、尺度和载荷必须原样用于新数据；若在全数据上重新标准化再评估，会产生信息泄漏。
<!-- bilingual-en:start -->
PCA is sensitive to outliers and heavy tails because the covariance matrix is sensitive. The centre, scale, and loadings estimated on a training set must be applied unchanged to new data. Re-standardising on the full dataset before evaluation creates information leakage.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### PCA 与因子分析最根本的问题差别是什么？
<!-- bilingual-en:start -->
*What is the most fundamental difference between the questions asked by PCA and factor analysis?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> PCA 问怎样用线性组合保留最多总变异；因子分析问哪些潜在公共因子能解释变量间协方差。
> <!-- bilingual-en:start -->
> PCA asks which linear combinations retain the most total variation. Factor analysis asks which latent common factors explain covariance among observed variables.
> <!-- bilingual-en:end -->

### 什么时候应优先基于相关矩阵做 PCA？
<!-- bilingual-en:start -->
*When should PCA based on a correlation matrix be preferred?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 变量量纲或自然尺度差异很大、且不希望大方差单位自动主导结果时。
> <!-- bilingual-en:start -->
> When variables have very different units or natural scales and large-variance units should not automatically dominate the result.
> <!-- bilingual-en:end -->

### 累计解释率达到 90% 是否自动说明选出的成分足够？
<!-- bilingual-en:start -->
*Does reaching 90% cumulative explained variance automatically mean that enough components have been retained?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不自动。阈值未考虑下游用途、小方差但重要的信号、样本稳定性和新数据表现，需要结合任务验证。
> <!-- bilingual-en:start -->
> No. The threshold ignores downstream purpose, low-variance but important signals, sampling stability, and performance on new data. It must be combined with task-specific validation.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Penn State STAT 505](https://online.stat.psu.edu/stat505/)：核验多元正态、均值推断、MANOVA、PCA、因子分析、判别与聚类的定义和使用条件。
- Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验矩阵公式与抽样分布。
<!-- bilingual-en:start -->
- [Penn State STAT 505](https://online.stat.psu.edu/stat505/) was used to verify the definitions and conditions of the multivariate methods in this course.
- Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was used to verify matrix formulas and sampling distributions.
<!-- bilingual-en:end -->
- [Penn State STAT 505, Lesson 11](https://online.stat.psu.edu/stat505/Lesson11)：逐项核验方差最大化、特征值/特征向量、标准化、成分选择与解释流程。
<!-- bilingual-en:start -->
- [Penn State STAT 505, Lesson 11](https://online.stat.psu.edu/stat505/Lesson11) was checked section by section for variance maximisation, eigenvalues and eigenvectors, standardisation, component selection, and interpretation.
<!-- bilingual-en:end -->
