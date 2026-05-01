---
aliases:
- Panel Data Model
- 面板数据模型
- panel data
tags:
- concept
- econometrics
---
# Panel Data Model

## 先记一句话

面板数据同时有个体维度和时间维度，可以利用“同一个体随时间变化”和“不同个体差异”估计模型。

## 它是什么

典型形式：

$$
y_{it}=\alpha_i+\lambda_t+x_{it}'\beta+u_{it}
$$

其中 $\alpha_i$ 是个体效应，$\lambda_t$ 是时间效应。

## 解决什么判断

它回答：“数据是否能用个体和时间的双重结构控制不可观测异质性？”

## 最小例子

跟踪 30 个省份 10 年的 GDP、教育支出和产业结构，就是省份-年份面板数据。

## 易混点

- 面板数据不自动解决内生性，只能帮助控制某些不随时间变化的遗漏变量。
- [[Fixed Effects Model]] 允许个体效应和解释变量相关；[[Random Effects Model]] 要求二者不相关。
- 政策评估中的 DID 常用双向固定效应实现，但识别靠 [[Parallel Trends]]。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[Fixed Effects Model]]
- [[Random Effects Model]]
- [[DID]]
- [[Hausman Test]]
