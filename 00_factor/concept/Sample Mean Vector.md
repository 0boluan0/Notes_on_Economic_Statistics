---
aliases:
- Sample Mean Vector
- 样本均值向量
tags:
- concept
- multivariate statistics
---
# Sample Mean Vector

>[!note] 一句话记忆
> 样本均值向量是总体均值向量的样本估计，把每个变量的样本平均值按列排列。

## 它是什么

对样本 $X_1,\ldots,X_n$，
$$
\bar X=\frac1n\sum_{j=1}^n X_j.
$$

若写成分量：
$$
\bar X=(\bar X_1,\ldots,\bar X_p)'.
$$

## 解决什么判断

- 样本中心在哪里。
- Hotelling $T^2$ 中观测中心离目标中心有多远。
- 两组或多组均值向量比较的基础统计量是什么。

## 最小例子

三变量样本的平均身高、平均体重、平均血压组成一个三维样本均值向量。

## 易混点

- 样本均值向量不是一行原始观测，而是所有样本的平均中心。
- 推断时要同时看 $\bar X$ 和 [[Sample Covariance Matrix]]。

## 来自课程位置

- [[01_introduction简介#1.4. 基本描述统计矩阵|第1章 基本描述统计矩阵]]

## 关联卡片

- [[Mean Vector]]
- [[Sample Covariance Matrix]]
- [[Hotelling T2 Test]]
