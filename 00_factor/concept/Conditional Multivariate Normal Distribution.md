---
aliases:
- Conditional Multivariate Normal Distribution
- Conditional Normal Distribution
- 多元正态条件分布
- 条件多元正态分布
tags:
- concept
- multivariate statistics
---
# Conditional Multivariate Normal Distribution

>[!note] 一句话记忆
> 多元正态的条件分布仍然是正态，条件均值会按协方差结构线性修正。

## 它是什么

设
$$
\begin{bmatrix}X_1\\X_2\end{bmatrix}
\sim N\left(
\begin{bmatrix}\mu_1\\\mu_2\end{bmatrix},
\begin{bmatrix}
\Sigma_{11}&\Sigma_{12}\\
\Sigma_{21}&\Sigma_{22}
\end{bmatrix}
\right).
$$

则
$$
X_1\mid X_2=x_2
\sim N\left(
\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),
\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\right).
$$

## 解决什么判断

- 已知一部分变量后，另一部分变量的分布如何更新。
- 协方差结构如何影响条件均值。
- 为什么正态模型在预测和判别中方便。

## 最小例子

已知某人的身高后，体重的条件均值会从总体平均体重向与身高相关的方向调整。

## 易混点

- 条件协方差不依赖观测值 $x_2$，只依赖 $\Sigma$ 的分块。
- 条件均值依赖 $x_2-\mu_2$。
- 课程里通常只需会识别公式，不一定要求完整推导。

## 来自课程位置

- [[04_多元正态分布The Multivariate Normal Distribution#1.3. 多元正态分布的性质|第4章 3 多元正态性质]]

## 关联卡片

- [[Multivariate Normal Distribution]]
- [[Covariance Matrix]]
- [[Matrix Inverse]]
