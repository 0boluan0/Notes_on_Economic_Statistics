---
aliases:
- Generalized Least Squares
- GLS
- 广义最小二乘
tags:
- concept
- econometrics
---
# Generalized Least Squares

## 先记一句话

GLS 在误差协方差不是 $\sigma^2I$ 时，用协方差结构重新加权模型以提高效率。

## 它是什么

若：

$$
\operatorname{Var}(u\mid X)=\Omega
$$

GLS 估计量为：

$$
\hat\beta_{GLS}=(X'\Omega^{-1}X)^{-1}X'\Omega^{-1}y
$$

## 解决什么判断

它回答：“误差存在异方差或自相关时，能否利用误差协方差结构改进估计？”

## 最小例子

若误差满足 AR(1)，可对数据做准差分，再用 OLS 估计变换后的模型。

## 易混点

- [[Weighted Least Squares]] 是 GLS 在对角协方差矩阵下的特例。
- $\Omega$ 未知时用 [[FGLS]]。
- 若只想修正标准误，不一定要改系数；可用 [[White Robust Standard Errors]] 或 [[Newey-West]]。

## 来自课程位置

- [[07_异方差]]
- [[08_自相关]]

## 关联卡片

- [[Weighted Least Squares]]
- [[FGLS]]
- [[Cochrane-Orcutt]]
- [[Newey-West]]
