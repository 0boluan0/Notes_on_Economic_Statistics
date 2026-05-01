---
aliases:
- Omitted Variable Bias
- OVB
- 遗漏变量偏误
tags:
- concept
- econometrics
---
# Omitted Variable Bias

## 先记一句话

遗漏变量偏误来自“遗漏变量既影响 $Y$，又和保留的解释变量相关”。

## 它是什么

真实模型：

$$
Y_i=\beta_0+\beta_1X_{1i}+\beta_2X_{2i}+u_i
$$

若估计时遗漏 $X_2$，则 $\hat\beta_1$ 的偏误方向为：

$$
\operatorname{Bias}(\hat\beta_1)
=\beta_2\frac{\operatorname{Cov}(X_1,X_2)}{\operatorname{Var}(X_1)}
$$

## 解决什么判断

它回答：“少放一个变量会不会让保留变量的系数系统性错误？”

## 最小例子

估计教育对工资的影响时遗漏能力。如果能力提高工资，且能力与教育正相关，则教育系数向上偏。

## 易混点

- 只要遗漏变量和 $X_1$ 不相关，就不一定造成 $\hat\beta_1$ 偏误。
- 加入无关变量通常影响效率；遗漏相关变量会影响一致性。
- OVB 是 [[Endogeneity]] 的常见来源。

## 来自课程位置

- [[04_模型设定]]

## 关联卡片

- [[Model Misspecification]]
- [[Endogeneity]]
- [[Measurement Error]]
