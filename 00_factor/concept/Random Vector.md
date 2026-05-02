---
aliases:
- Random Vector
- 随机向量
tags:
- concept
- multivariate statistics
---
# Random Vector

>[!note] 一句话记忆
> 随机向量是把多个随机变量按列放在一起，用一个对象同时描述多个变量的随机性。

## 它是什么

$$
X=
\begin{bmatrix}
X_1\\
X_2\\
\vdots\\
X_p
\end{bmatrix}.
$$

每个分量 $X_i$ 是一个随机变量，整个 $X$ 的分布称为联合分布。

## 解决什么判断

- 数据是否需要多变量方法，而不是对每个变量分别分析。
- 均值、方差和协方差是否要用向量或矩阵表达。
- 线性组合 $a'X$ 的均值与方差如何计算。

## 最小例子

一个学生的数学、英语、统计成绩可写成三维随机向量 $X=(X_1,X_2,X_3)'$。

## 易混点

- 随机向量不是普通数据行；它是一个随机对象。
- $X_i$ 可以相关，不能默认彼此独立。
- 多元统计的核心不是变量多，而是变量之间的协方差结构也重要。

## 来自课程位置

- [[02_矩阵代数和随机向量Matrix Algebra and Random Vectors#1.7. 随机向量与随机矩阵|第2章 7 随机向量与随机矩阵]]

## 关联卡片

- [[Mean Vector]]
- [[Covariance Matrix]]
- [[Correlation Matrix]]
- [[Multivariate Normal Distribution]]
