---
aliases:
  - "概率 PCA 假定观测由低维 Gaussian 潜变量与各向同性 Gaussian 噪声线性生成"
  - Probabilistic PCA
  - PPCA
student_os: knowledge-atom
atom_id: STAT-PCA-019
atom_set: principal-component-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[主成分分析.canvas|主成分分析]]"
requires:
  - "[[主成分分析]]"
  - "[[联合Gaussian]]"
contrasts_with:
  - "[[公共因子模型]]"
related:
  - "[[PCA与因子分析]]"
  - "[[主成分不等于潜变量]]"
---

# 概率 PCA 假定观测由低维 Gaussian 潜变量与各向同性 Gaussian 噪声线性生成
<!-- bilingual-en:start -->
*Probabilistic PCA assumes that observations are generated linearly from lower-dimensional Gaussian latent variables plus isotropic Gaussian noise*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 概率主成分分析（probabilistic PCA, PPCA）使用生成模型
> $$
> X=\mu+WZ+\varepsilon,\qquad Z\sim N(0,I_q),\qquad \varepsilon\sim N(0,\sigma^2I_p),
> $$
> 并假定 $Z$ 与 $\varepsilon$ 独立。因此 $X\sim N(\mu,WW^T+\sigma^2I_p)$。它是一个明确的概率潜变量模型，不是普通 PCA 定义中隐含的生成假设。
> <!-- bilingual-en:start -->
> PPCA is a latent Gaussian model with covariance $WW^T+\sigma^2I$. Classical PCA is an algebraic transformation and does not by itself make this generative assumption.
> <!-- bilingual-en:end -->

当潜维数 $q<p$ 固定并用极大似然拟合时，$W$ 的列空间由样本协方差的前 $q$ 个特征向量张成，因而与普通 PCA 的主子空间对应。但这种对应只说明最优子空间相同，不说明两种方法的模型语义相同。

概率 PCA 的噪声协方差是 $\sigma^2I_p$：每个观测变量都有同一个噪声方差，且噪声间不相关。[[公共因子模型|公共因子模型]]通常允许对角噪声协方差 $\Psi$，即不同变量有不同的特殊方差。这一差异改变了拟合目标和载荷解释，不能只因为都出现潜变量就把它们当成同一模型。

即使 PPCA 写下了潜变量 $Z$，估计的潜空间仍有旋转不唯一：对任意正交矩阵 $R$，$WR$ 与相应旋转后的潜坐标给出同一个 $WW^T$。因此将某一列 $W$ 命名为真实构念，仍需理论、识别约束与外部证据；见[[主成分不等于潜变量]]。

> [!question]- 自检
> 概率 PCA 与普通 PCA 的主子空间在极大似然解中对应，是否意味着普通 PCA 已经假定了 Gaussian 潜变量生成模型？
>
> **答案：** 不意味。普通 PCA 可以作为协方差特征分解的代数方法使用；PPCA 额外提出了潜变量、Gaussian 分布和各向同性噪声假设。

## 来源与核验

- M. E. Tipping and C. M. Bishop, [Probabilistic Principal Component Analysis](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ppca-jrss.pdf)：核对 PPCA 的生成模型、边际协方差、极大似然主子空间与旋转不唯一。
- [scikit-learn, Exact PCA and probabilistic interpretation](https://scikit-learn.org/stable/modules/decomposition.html#exact-pca-and-probabilistic-interpretation)：核对普通 PCA 与概率 PCA 的对应及各向同性噪声假设。
- [[PCA与因子分析]] 与 [[公共因子模型]]：对照压缩、概率潜变量模型与逐变量特殊方差三种语义。
