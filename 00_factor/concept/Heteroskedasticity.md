---
aliases:
- Heteroskedasticity
- Heteroscedasticity
- 异方差
- 异方差性
tags:
  - concept
  - econometrics
---
# Heteroskedasticity

## 先记一句话

异方差是误差项方差随观测对象或解释变量变化，而不是保持常数。

## 它是什么

同方差假设要求：

$$
Var(u_i\mid X)=\sigma^2
$$

异方差时：

$$
Var(u_i\mid X)=\sigma_i^2
$$

## 解决什么判断

它回答：“OLS 的残差波动是不是在某些样本或变量取值下更大？”

## 最小例子

收入越高的家庭消费金额波动越大，因此消费函数的误差方差可能随收入增加。

## 易混点

- 在外生性仍成立时，异方差通常不破坏 OLS 无偏性和一致性，但会破坏经典标准误和 BLUE 有效性。
- 异方差诊断见 [[Heteroscedasticity Diagnosis]]；具体 White 检验步骤见 [[White Test Steps]]。
- 不知道方差形式时常用 [[White Robust Standard Errors]]；知道或能可靠估计方差形式时可用 [[Weighted Least Squares]]。

## 来自课程位置

- [[07_异方差]]

## 关联卡片

- [[White Test]]
- [[White Robust Standard Errors]]
- [[Weighted Least Squares]]
- [[Gauss-Markov theorem]]
