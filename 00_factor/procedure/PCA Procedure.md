---
aliases:
- PCA Procedure
- Principal Component Analysis Steps
- 主成分分析步骤
tags:
- procedure
- multivariate statistics
---
# PCA Procedure

## 输入

- 数据矩阵 $X$。
- 选择协方差矩阵 $S$ 或相关矩阵 $R$。

## 输出

- 主成分方向。
- 方差解释率。
- 保留的主成分数量。

## Step 1. 选择矩阵

- 变量量纲相近：可用 $S$。
- 变量量纲差异大：优先用 $R$。

## Step 2. 求特征值和特征向量

$$
Se_i=\lambda_i e_i
$$
或
$$
Re_i=\lambda_i e_i.
$$

按 $\lambda_1\geq\cdots\geq\lambda_p$ 排序。

## Step 3. 构造主成分

$$
Y_i=e_i'X.
$$

样本中用中心化或标准化后的观测代入。

## Step 4. 计算解释率

$$
\text{PVE}_i=\frac{\lambda_i}{\sum_{j=1}^p\lambda_j}.
$$

前 $m$ 个累计解释率为
$$
\frac{\sum_{i=1}^m\lambda_i}{\sum_{j=1}^p\lambda_j}.
$$

## Step 5. 选择主成分数量

结合：

- 累计方差解释率；
- [[Scree Plot]]；
- 变量解释性。

## 检查点

- 是否已中心化或标准化。
- 特征向量方向符号可反，不影响主成分本质。
- 保留数量不能只看一个机械阈值。

## 来自课程位置

- [[08_主成分分析principal component#1.2. 总体主成分（Population Principal Components）|第8章 2 总体主成分]]

## 关联卡片

- [[PCA]]
- [[Variance Explained]]
- [[Choosing Covariance vs Correlation Matrix]]
