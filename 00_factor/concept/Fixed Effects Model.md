---
aliases:
- Fixed Effects Model
- Fixed Effects
- 固定效应模型
- 固定效应
tags:
- concept
- econometrics
---

# Fixed Effects Model

## 先记一句话

固定效应用个体或时间自己的截距，吸收不随时间或不随个体变化的遗漏因素。

## 它是什么

个体固定效应模型：

$$
y_{it}=\alpha_i+x_{it}'\beta+u_{it}
$$

双向固定效应：

$$
y_{it}=\alpha_i+\lambda_t+x_{it}'\beta+u_{it}
$$

## 解决什么判断

它回答：“有没有不随时间变化的个体特质会同时影响 $x$ 和 $y$，需要被控制掉？”

## 最小例子

研究企业研发投入对产出的影响时，企业管理能力可能长期稳定且影响研发和产出。企业固定效应可吸收这类不随时间变化的能力差异。

## 易混点

- 固定效应只能控制不随时间变化的遗漏变量；时间变化的遗漏冲击仍可能造成内生性。
- 固定效应不能估计不随时间变化变量的系数。
- DID 常用固定效应回归，但 DID 的关键假设是 [[Parallel Trends]]。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[Panel Data Model]]
- [[Random Effects Model]]
- [[Hausman Test]]
- [[DID]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Parallel Trends]]、[[13_面板数据模型]]、[[Panel Data Model]]、[[Random Effects Model]]、[[Hausman Test]]、[[DID]]。
