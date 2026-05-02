# 1. 第8章：主成分分析（Principal Component Analysis）

>[!summary] 本章主线
> PCA 的核心是把原变量旋转成一组互不相关的新变量，并按方差大小排序。它回答的是“哪些方向保留了最多信息”。

## 1.1. PCA 的目标

主成分分析（PCA）主要用于：

1. 数据降维。
2. 用少数线性组合解释大部分变异。
3. 发现主要变异方向。
4. 在变量高度相关时构造互不相关的新指标。

>[!note] 一句话
> PCA 不是找最重要的原变量，而是找最重要的线性组合。

## 1.2. 总体主成分（Population Principal Components）

给定随机向量
$$
X=(X_1,\ldots,X_p)'
$$
具有均值 $\mu$ 和协方差矩阵 $\Sigma$。

令
$$
\Sigma e_i=\lambda_i e_i,\qquad
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_p.
$$

第 $i$ 个主成分定义为
$$
Y_i=e_i'X.
$$

其性质为
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

总体总变异为
$$
\operatorname{tr}(\Sigma)=\sum_{i=1}^p\lambda_i.
$$

第 $k$ 个主成分的方差解释率为
$$
\frac{\lambda_k}{\sum_{i=1}^p\lambda_i}.
$$

前 $m$ 个主成分的累计解释率为
$$
\frac{\sum_{i=1}^m\lambda_i}{\sum_{i=1}^p\lambda_i}.
$$

>[!example] 课后题提示
> 课后题出现过 $\rho_{Y_iZ_j}=w_{ij}\sqrt{\lambda_i}$ 这一类主成分和标准化变量之间相关性的表达。复习时把它理解为“载荷/相关性由特征向量元素和特征值共同决定”。

## 1.4. 标准化变量的主成分

如果变量量纲差异大，使用协方差矩阵会让高方差变量主导 PCA。

此时可先标准化：
$$
Z_j=\frac{X_j-\mu_j}{\sqrt{\sigma_{jj}}},
$$
再对相关矩阵 $\rho$ 做特征值分解。

标准化变量的主成分为
$$
Y_i=e_i'Z.
$$

>[!tip] 判断
> 协方差矩阵 PCA 保留原始尺度；相关矩阵 PCA 相当于先让每个变量方差为 1。

## 1.5. 样本主成分

实际计算中用样本协方差矩阵 $S$ 代替 $\Sigma$：
$$
S=\frac{1}{n-1}D'D.
$$

对 $S$ 求特征值和特征向量：
$$
Se_i=\hat\lambda_i e_i.
$$

样本主成分得分由中心化后的样本代入 $e_i'X$ 得到。

## 1.6. 主成分数量选择

常用标准：

1. 累计方差解释率达到目标阈值。
2. 碎石图出现明显拐点。
3. 保留后的主成分仍有可解释意义。

>[!warning] 不要机械化
> “累计解释率超过 80%”只是经验规则。考试和实务中都要结合题目要求、变量含义和碎石图。

## 1.7. 大样本性质

在正态等正则条件下，样本特征值具有渐近正态性质：
$$
\sqrt n(\hat\lambda_i-\lambda_i)
\overset{a}{\sim}
N(0,2\lambda_i^2).
$$

近似置信区间可写作
$$
\hat\lambda_i\pm z_{\alpha/2}\sqrt{\frac{2\lambda_i^2}{n}}.
$$

>[!note] 考试提示
> 旧笔记标注“大样本性质不考”。复习时优先掌握 PCA 的定义、方差解释率、协方差矩阵 vs 相关矩阵。

## 1.8. 关联卡片

- [[PCA]]
- [[PCA Procedure]]
- [[Variance Explained]]
- [[Scree Plot]]
- [[Choosing Covariance vs Correlation Matrix]]
- [[PCA vs Factor Analysis]]
