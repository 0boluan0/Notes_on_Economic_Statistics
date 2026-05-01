---
aliases:
- Credit VaR
- Credit Value at Risk
- 信用VaR
- 信用在险价值
tags:
- concept
- credit-risk
- risk-management
---
# Credit VaR

## 先记一句话

Credit VaR 是信用组合损失分布在高置信水平下的分位损失。

## 它是什么

若信用损失为 $L$：

$$
Credit\ VaR_\alpha=q_\alpha(L)
$$

常见管理口径会关注非预期损失，即 Credit VaR 减去预期损失：

$$
UL_\alpha=Credit\ VaR_\alpha-EL
$$

其中：

$$
EL=PD\times LGD\times EAD
$$

## 解决什么判断

它回答：“在极端但给定置信水平内，信用组合可能亏到多少？”

## 最小例子

99.9% Credit VaR = 5 亿，预期损失 = 1 亿，则非预期损失为 4 亿，可作为经济资本参考。

## 易混点

- 市场 [[VaR]] 由市场价格波动驱动；Credit VaR 由违约、评级迁移和相关性驱动。
- 信用损失分布通常偏斜、厚尾，不应机械套正态。
- 组合相关性会显著影响尾部损失。

## 来自课程位置

- [[21_信用在险价值]]

## 关联卡片

- [[Credit Risk]]
- [[PD]]
- [[LGD]]
- [[EAD]]
- [[Basel Accords]]
