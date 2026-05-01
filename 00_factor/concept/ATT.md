---
aliases:
- ATT
- Average Treatment Effect on the Treated
- 处理组平均处理效应
tags:
- concept
- econometrics
- causal-inference
---
# ATT

## 先记一句话

ATT 是处理真正发生在处理组身上时，对处理组产生的平均因果效应。

## 它是什么

$$
ATT=E[Y(1)-Y(0)\mid G=1]
$$

其中 $Y(1)$ 是接受处理的潜在结果，$Y(0)$ 是未接受处理的潜在结果。

## 解决什么判断

它回答：“政策对实际被处理的人或地区平均产生了多大影响？”

## 最小例子

只关心最低工资政策对实施省份的影响，而不是对所有省份如果都实施会怎样，这就是 ATT 口径。

## 易混点

- ATT 不等于 ATE；ATE 是全体样本平均处理效应。
- 标准两组两期 DID 在平行趋势下识别 ATT。
- 估计 ATT 的可信度取决于反事实 $Y(0)$ 是否能由对照组趋势代表。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[DID]]
- [[Parallel Trends]]
- [[DID Identification Proof]]
