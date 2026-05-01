---
aliases:
- Residual
- 残差
- OLS residual
tags:
- concept
- econometrics
---
# Residual

## 先记一句话

Residual 是样本中实际值和模型拟合值之间的差。

## 它是什么

$$
\hat u_i=y_i-\hat y_i
$$

矩阵形式：

$$
\hat u=y-X\hat\beta
$$

## 解决什么判断

它回答：“模型在每个样本点上还剩下多少没有解释的部分？”

## 最小例子

实际工资为 5000，模型预测工资为 4600，则残差为 400。

## 易混点

- 残差 $\hat u_i$ 是估计后可观察的；误差项 $u_i$ 是真实但不可观察的。
- OLS 残差与解释变量正交：$X'\hat u=0$。
- 残差图常用于诊断 [[Heteroskedasticity]]、[[Autocorrelation]] 和 [[Model Misspecification]]。

## 来自课程位置

- [[02_一元线性回归]]
- [[03_多元线性回归]]

## 关联卡片

- [[OLS Basics]]
- [[OLS Normal Equations]]
- [[R-squared]]
- [[Outlier Detection]]
