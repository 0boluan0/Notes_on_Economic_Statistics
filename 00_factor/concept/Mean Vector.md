---
aliases:
- Mean Vector
- Population Mean Vector
- 均值向量
- 总体均值向量
tags:
- concept
- multivariate statistics
---
# Mean Vector

>[!note] 一句话记忆
> 均值向量把每个变量的总体均值按同一顺序排列，是多元分布的中心。

## 它是什么

对随机向量 $X=(X_1,\ldots,X_p)'$，
$$
\mu=E(X)=
\begin{bmatrix}
E(X_1)\\
\vdots\\
E(X_p)
\end{bmatrix}.
$$

## 解决什么判断

- 多元分布的中心在哪里。
- 均值向量是否等于目标值 $\mu_0$。
- 两组或多组数据的中心是否有显著差异。

## 最小例子

若 $X=(收益率, 波动率)'$，$\mu=(0.08,0.20)'$ 表示长期平均收益率为 8%，平均波动率为 20%。

## 易混点

- 均值向量只描述中心，不描述变量间相关性。
- 多元检验通常检验整个向量，而不是逐个分量单独检验。

## 来自课程位置

- [[01_introduction简介#1.4. 基本描述统计矩阵|第1章 基本描述统计矩阵]]
- [[05_ 总体平均向量的推论#1. 第5章：总体平均向量的推论（Inferences about Population Mean Vector）|第5章 总体平均向量推断]]

## 关联卡片

- [[Sample Mean Vector]]
- [[Hotelling T2 Test]]
- [[Multivariate Mean Inference Map]]
