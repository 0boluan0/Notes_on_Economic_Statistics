---
aliases:
- Random Effects Model
- Random Effects
- 随机效应模型
- 随机效应
tags:
- concept
- econometrics
---
# Random Effects Model

## 先记一句话

随机效应把个体效应当作与解释变量不相关的随机误差成分，并用 GLS 提高效率。

## 它是什么

$$
y_{it}=\beta_0+x_{it}'\beta+\alpha_i+u_{it}
$$

核心假设：

$$
Cov(\alpha_i,x_{it})=0
$$

## 解决什么判断

它回答：“个体异质性是否可以当作外生随机成分，而不是和解释变量系统相关？”

## 最小例子

若抽样个体来自总体，且个体差异与解释变量无关，可以考虑随机效应以同时利用组内和组间信息。

## 易混点

- 如果 $\alpha_i$ 与 $x_{it}$ 相关，随机效应不一致，应使用 [[Fixed Effects Model]]。
- 随机效应可以估计不随时间变化变量的系数，固定效应通常不能。
- FE vs RE 常用 [[Hausman Test]] 辅助判断，但最终仍要靠理论。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[Panel Data Model]]
- [[Fixed Effects Model]]
- [[Hausman Test]]
- [[FGLS]]
