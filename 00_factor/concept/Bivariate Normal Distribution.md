---
aliases:
- Bivariate Normal Distribution
- 二元正态分布
- 二维正态分布
tags:
- concept
- multivariate statistics
---

# Bivariate Normal Distribution

>[!note] 一句话记忆
> 二元正态分布是多元正态分布在 $p=2$ 时的特例，用两个变量的均值、方差和相关系数描述联合分布。

## 它是什么

若
$$
\begin{bmatrix}X\\Y\end{bmatrix}
\sim N_2\left(
\begin{bmatrix}\mu_X\\\mu_Y\end{bmatrix},
\begin{bmatrix}
\sigma_X^2&\rho\sigma_X\sigma_Y\\
\rho\sigma_X\sigma_Y&\sigma_Y^2
\end{bmatrix}
\right),
$$
则 $(X,Y)$ 服从二元正态分布。

## 解决什么判断

- 两个正态变量的联合密度如何写。
- 相关系数 $\rho$ 如何改变等密度椭圆的倾斜和形状。
- 二维多元正态公式如何展开。

## 最小例子

若 $\rho=0$，协方差矩阵为对角矩阵，等密度曲线的主轴与坐标轴对齐。

## 易混点

- 边际正态不保证联合二元正态。
- 在二元正态下，$\rho=0$ 才能推出独立；一般分布不成立。

## 来自课程位置

- [[04_多元正态分布The Multivariate Normal Distribution#1.2. 多元正态密度及等密度曲线|第4章 2 多元正态密度]]

## 关联卡片

- [[Multivariate Normal Distribution]]
- [[Correlation Matrix]]
- [[Mahalanobis Distance]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[04_多元正态分布The Multivariate Normal Distribution]]、[[Multivariate Normal Distribution]]、[[Correlation Matrix]]、[[Mahalanobis Distance]]。
