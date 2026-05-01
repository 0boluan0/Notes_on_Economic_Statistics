---
aliases:
- Weighted Least Squares
- WLS
- Generalized Least Squares
- GLS
- 加权最小二乘法
- 加权最小二乘法(WLS)
tags:
- concept
- econometrics
---
# Weighted Least Squares

## 先记一句话

WLS 通过给不同观测赋权，把异方差模型变换成更接近同方差的模型。

## 它是什么

若：

$$
Var(u_i\mid X)=\sigma_i^2
$$

理想权重与方差成反比：

$$
w_i=\frac{1}{\sigma_i^2}
$$

WLS 最小化：

$$
\min_\beta \sum_i w_i(y_i-x_i'\beta)^2
$$

矩阵形式：

$$
\hat\beta_{WLS}=(X'WX)^{-1}X'Wy
$$

## 解决什么判断

它回答：“如果不同观测误差方差不同，能否通过权重提高估计效率？”

## 最小例子

大企业销售额波动更大。若误差方差与企业规模成比例，可以给大企业观测较低权重，降低高方差观测对估计的影响。

## 易混点

- WLS 改变系数估计；[[White Robust Standard Errors]] 只改变标准误。
- 权重设错可能比不用权重更差。
- 不知道真实方差时，用估计权重的版本叫 [[FGLS]]，步骤见 [[Weighted Least Squares Estimation]]。

## 来自课程位置

- [[07_异方差]]

## 关联卡片

- [[Heteroskedasticity]]
- [[FGLS]]
- [[White Robust Standard Errors]]
- [[OLS Basics]]
