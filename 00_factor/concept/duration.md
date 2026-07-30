---
aliases:
- Duration
- duration
- 久期
- 持续期
tags:
- concept
- fixed-income
- risk-management
---

# Duration

## 先记一句话

Duration 是固定收益工具现金流时间和利率敏感度的入口：现金流越晚回来，价格通常越怕利率上升。

## 它是什么

最基础的久期是未来现金流现值权重下的平均回收时间：

$$
D=\sum_{t=1}^{n}t\frac{PV(CF_t)}{P}
$$

其中 $P=\sum_t PV(CF_t)$ 是债券价格。

## 解决什么判断

它回答两个问题：

- 现金流平均多晚收回。
- 利率小幅变化时，价格大约会变动多少。

## 最小例子

零息债券只有到期一次现金流，所以它的 Macaulay duration 等于剩余期限。附息债券中间有票息现金流，所以 duration 通常短于到期期限。

## 易混点

- [[Macaulay Duration]] 是时间口径；[[Modified Duration]] 是价格敏感度口径。
- [[Dollar Duration]] 和 [[Basis Point Value (BPV)]] 是金额口径。
- 利率变化较大时，单用久期会有线性近似误差，需要 [[Convexity]]。
- 含提前还款或可赎回条款的债券，现金流会变，应看 [[Effective Duration]]。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Macaulay Duration]]
- [[Modified Duration]]
- [[Dollar Duration]]
- [[Effective Duration]]
- [[Key Rate Duration]]
- [[Convexity]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Macaulay Duration]]、[[Modified Duration]]、[[Dollar Duration]]、[[Basis Point Value (BPV)]]、[[Convexity]]、[[Effective Duration]]、[[09_利率风险]]、[[Key Rate Duration]]。
