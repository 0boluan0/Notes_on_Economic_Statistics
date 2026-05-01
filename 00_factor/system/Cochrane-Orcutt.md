---
aliases:
- Cochrane-Orcutt
- Cochrane-Orcutt Iteration
- CO迭代
tags:
- system
- econometrics
---
# Cochrane-Orcutt

## 诊断目标

Cochrane-Orcutt 在误差服从 AR(1) 且结构可信时，通过估计 $\rho$ 并广义差分来提高估计效率。

## 模型前提

$$
u_t=\rho u_{t-1}+e_t
$$

且 $|\rho|<1$。

## 流程

1. 先用 OLS 得到残差 $\hat u_t$。
2. 用 $\hat u_t$ 对 $\hat u_{t-1}$ 回归估计 $\hat\rho$。
3. 做变换：

$$
y_t^*=y_t-\hat\rho y_{t-1},\qquad x_t^*=x_t-\hat\rho x_{t-1}
$$

4. 对变换后的模型做 OLS。
5. 迭代直到 $\hat\rho$ 和系数稳定。

## 适用边界

- 适合一阶自相关结构明确的情形。
- 会损失第一期观测；小样本可考虑 Prais-Winsten。
- 若自相关来自模型遗漏，不应只做机械变换。

## 常见错误

- 未确认 AR(1) 就套用。
- $\rho$ 接近 1 时忽略单位根问题。
- 只追求 DW 接近 2，不解释模型经济含义。

## 来自课程位置

- [[08_自相关]]

## 关联卡片

- [[Autocorrelation]]
- [[Autocorrelation Diagnosis]]
- [[Newey-West]]
- [[FGLS]]
