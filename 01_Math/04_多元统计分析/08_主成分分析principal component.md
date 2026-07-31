# 1. 第8章：主成分分析（Principal Component Analysis）
<!-- bilingual-en:start -->
*1. Chapter 8: Principal Component Analysis*
<!-- bilingual-en:end -->

>[!note] 本章主线
> PCA 的核心是把原变量旋转成一组互不相关的新变量，并按方差大小排序。它回答的是“哪些方向保留了最多信息”。
> <!-- bilingual-en:start -->
> PCA rotates the original variables into a set of uncorrelated new variables and orders them by variance. It asks which directions preserve the most information.
> <!-- bilingual-en:end -->

## 1.1. PCA 的目标
<!-- bilingual-en:start -->
*1.1. The Aim of PCA*
<!-- bilingual-en:end -->

主成分分析（PCA）主要用于：
<!-- bilingual-en:start -->
Principal component analysis is mainly used to:
<!-- bilingual-en:end -->

1. 数据降维。
2. 用少数线性组合解释大部分变异。
3. 发现主要变异方向。
4. 在变量高度相关时构造互不相关的新指标。
<!-- bilingual-en:start -->
1. Reduce the dimensionality of data.
2. Explain most variation with a small number of linear combinations.
3. Discover the main directions of variation.
4. Construct uncorrelated new indices when the original variables are highly correlated.
<!-- bilingual-en:end -->

>[!note] 一句话
> PCA 不是找最重要的原变量，而是找最重要的线性组合。
> <!-- bilingual-en:start -->
> PCA does not identify the most important original variable; it identifies the most important linear combinations.
> <!-- bilingual-en:end -->

## 1.2. 总体主成分（Population Principal Components）
<!-- bilingual-en:start -->
*1.2. Population Principal Components*
<!-- bilingual-en:end -->

给定随机向量
<!-- bilingual-en:start -->
Let the random vector
<!-- bilingual-en:end -->
$$
X=(X_1,\ldots,X_p)'
$$
具有均值 $\mu$ 和协方差矩阵 $\Sigma$。
<!-- bilingual-en:start -->
have mean $\mu$ and covariance matrix $\Sigma$.
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->
$$
\Sigma e_i=\lambda_i e_i,\qquad
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_p.
$$

第 $i$ 个主成分定义为
<!-- bilingual-en:start -->
The $i$th principal component is defined as
<!-- bilingual-en:end -->
$$
Y_i=e_i'X.
$$

其性质为
<!-- bilingual-en:start -->
Its properties are
<!-- bilingual-en:end -->
$$
E(Y_i)=e_i'\mu,
$$
$$
\operatorname{Var}(Y_i)=e_i'\Sigma e_i=\lambda_i,
$$
$$
\operatorname{Cov}(Y_i,Y_k)=0,\quad i\neq k.
$$

## 1.3. 总变异与方差解释率
<!-- bilingual-en:start -->
*1.3. Total Variation and Explained-Variance Ratios*
<!-- bilingual-en:end -->

总体总变异为
<!-- bilingual-en:start -->
The total population variation is
<!-- bilingual-en:end -->
$$
\operatorname{tr}(\Sigma)=\sum_{i=1}^p\lambda_i.
$$

第 $k$ 个主成分的方差解释率为
<!-- bilingual-en:start -->
The proportion of variance explained by the $k$th principal component is
<!-- bilingual-en:end -->
$$
\frac{\lambda_k}{\sum_{i=1}^p\lambda_i}.
$$

前 $m$ 个主成分的累计解释率为
<!-- bilingual-en:start -->
The cumulative proportion explained by the first $m$ principal components is
<!-- bilingual-en:end -->
$$
\frac{\sum_{i=1}^m\lambda_i}{\sum_{i=1}^p\lambda_i}.
$$

>[!example] 课后题提示
> 课后题出现过 $\rho_{Y_iZ_j}=w_{ij}\sqrt{\lambda_i}$ 这一类主成分和标准化变量之间相关性的表达。复习时把它理解为“载荷/相关性由特征向量元素和特征值共同决定”。
> <!-- bilingual-en:start -->
> An exercise used an expression such as $\rho_{Y_iZ_j}=w_{ij}\sqrt{\lambda_i}$ for the correlation between a principal component and a standardised variable. Interpret it as saying that a loading or correlation is jointly determined by an eigenvector element and its eigenvalue.
> <!-- bilingual-en:end -->

## 1.4. 标准化变量的主成分
<!-- bilingual-en:start -->
*1.4. Principal Components of Standardised Variables*
<!-- bilingual-en:end -->

如果变量量纲差异大，使用协方差矩阵会让高方差变量主导 PCA。
<!-- bilingual-en:start -->
When variables have very different scales, covariance-based PCA allows high-variance variables to dominate.
<!-- bilingual-en:end -->

此时可先标准化：
<!-- bilingual-en:start -->
The variables can instead be standardised first:
<!-- bilingual-en:end -->
$$
Z_j=\frac{X_j-\mu_j}{\sqrt{\sigma_{jj}}},
$$
再对相关矩阵 $\rho$ 做特征值分解。
<!-- bilingual-en:start -->
Then perform an eigendecomposition of the correlation matrix $\rho$.
<!-- bilingual-en:end -->

标准化变量的主成分为
<!-- bilingual-en:start -->
The principal components of the standardised variables are
<!-- bilingual-en:end -->
$$
Y_i=e_i'Z.
$$

>[!item] 判断
> 协方差矩阵 PCA 保留原始尺度；相关矩阵 PCA 相当于先让每个变量方差为 1。
> <!-- bilingual-en:start -->
> Covariance-matrix PCA preserves the original scales; correlation-matrix PCA is equivalent to first giving every variable unit variance.
> <!-- bilingual-en:end -->

## 1.5. 样本主成分
<!-- bilingual-en:start -->
*1.5. Sample Principal Components*
<!-- bilingual-en:end -->

实际计算中用样本协方差矩阵 $S$ 代替 $\Sigma$：
<!-- bilingual-en:start -->
In practice, replace $\Sigma$ with the sample covariance matrix $S$:
<!-- bilingual-en:end -->
$$
S=\frac{1}{n-1}D'D.
$$

对 $S$ 求特征值和特征向量：
<!-- bilingual-en:start -->
Compute the eigenvalues and eigenvectors of $S$:
<!-- bilingual-en:end -->
$$
Se_i=\hat\lambda_i e_i.
$$

样本主成分得分由中心化后的样本代入 $e_i'X$ 得到。
<!-- bilingual-en:start -->
Sample principal-component scores are obtained by applying $e_i'X$ to the centred observations.
<!-- bilingual-en:end -->

## 1.6. 主成分数量选择
<!-- bilingual-en:start -->
*1.6. Choosing the Number of Principal Components*
<!-- bilingual-en:end -->

常用标准：
<!-- bilingual-en:start -->
Common criteria include:
<!-- bilingual-en:end -->

1. 累计方差解释率达到目标阈值。
2. 碎石图出现明显拐点。
3. 保留后的主成分仍有可解释意义。
<!-- bilingual-en:start -->
1. The cumulative explained variance reaches a target threshold.
2. The scree plot has a clear elbow.
3. The retained components remain interpretable.
<!-- bilingual-en:end -->

>[!attention] 不要机械化
> “累计解释率超过 80%”只是经验规则。考试和实务中都要结合题目要求、变量含义和碎石图。
> <!-- bilingual-en:start -->
> A cumulative explained variance above 80% is only a rule of thumb. In both examinations and practice, the decision should reflect the task, the meaning of the variables, and the scree plot.
> <!-- bilingual-en:end -->

## 1.7. 大样本性质
<!-- bilingual-en:start -->
*1.7. Large-Sample Properties*
<!-- bilingual-en:end -->

在正态等正则条件下，样本特征值具有渐近正态性质：
<!-- bilingual-en:start -->
Under normality and other regularity conditions, sample eigenvalues are asymptotically normal:
<!-- bilingual-en:end -->
$$
\sqrt n(\hat\lambda_i-\lambda_i)
\overset{a}{\sim}
N(0,2\lambda_i^2).
$$

近似置信区间可写作
<!-- bilingual-en:start -->
An approximate confidence interval can be written as
<!-- bilingual-en:end -->
$$
\hat\lambda_i\pm z_{\alpha/2}\sqrt{\frac{2\lambda_i^2}{n}}.
$$

>[!note] 考试提示
> 旧笔记标注“大样本性质不考”。复习时优先掌握 PCA 的定义、方差解释率、协方差矩阵 vs 相关矩阵。
> <!-- bilingual-en:start -->
> The old note says that large-sample properties are not examined. Prioritise the definition of PCA, explained-variance ratios, and the choice between covariance and correlation matrices.
> <!-- bilingual-en:end -->

## 1.8. 关联卡片
<!-- bilingual-en:start -->
*1.8. Related Cards*
<!-- bilingual-en:end -->

- [[主成分分析 PCA#PCA 的方差最大化|PCA]]
- [[主成分分析 PCA#PCA 的方差最大化|PCA Procedure]]
- [[主成分分析 PCA#标准化与成分选择|Variance Explained]]
- [[主成分分析 PCA#标准化与成分选择|Scree Plot]]
- [[主成分分析 PCA#标准化与成分选择|Choosing Covariance vs Correlation Matrix]]
- [[主成分分析 PCA#PCA 与因子分析的选择|PCA vs Factor Analysis]]
