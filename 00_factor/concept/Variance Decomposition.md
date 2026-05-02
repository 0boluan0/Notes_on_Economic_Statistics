---
aliases:
- Variance Decomposition
- Forecast Error Variance Decomposition
- FEVD
- 方差分解
- 预测误差方差分解
tags:
- concept
- 时间序列
---
# Variance Decomposition

## 先记一句话

方差分解就是：**把一个变量的预测误差方差拆成不同冲击来源的贡献比例**。

## 它是什么

在 VAR/VMA 中，预测误差来自多个结构冲击。方差分解问：

> 未来 $h$ 期的预测误差，有多少比例来自第 1 个冲击、第 2 个冲击、……？

## 它解决什么判断

- 哪个变量的冲击对目标变量预测误差最重要。
- 冲击贡献随预测期 $h$ 如何变化。
- 系统中信息传导的相对强弱。

## 最小例子

如果通胀的 12 期预测误差方差中，70% 来自自身冲击，20% 来自利率冲击，10% 来自产出冲击，那么通胀自身冲击是主要来源。

## 易混点

- 方差分解依赖冲击识别方式。
- Cholesky 方差分解受变量排序影响。
- 它解释的是预测误差方差贡献，不是变量水平的平均贡献。

## 来自课程位置

- [[05_多方程模型Multi-equation Time Series Models#4.4. 脉冲响应函数在VAR中的应用|时间序列 05：VMA 与冲击分析]]

## 关联卡片

- [[VAR Model]]
- [[VMA]]
- [[Impulse Response Function]]
- [[Structural VAR]]
