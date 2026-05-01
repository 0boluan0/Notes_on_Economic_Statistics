---
aliases:
- F-test diagnostics
- F统计量
- F检验诊断
tags:
- system
- econometrics
---
# F-test Diagnostics

## 诊断目标

这张 system 卡记录 F-test 在回归输出中如何解读，尤其是整体显著、联合限制和多重共线性的信号。

## 使用场景

- 检验模型整体是否有解释力。
- 检验一组变量是否联合显著。
- 比较受限模型与非受限模型。

## 诊断信号

| 现象 | 可能含义 |
| --- | --- |
| F 显著、多个 t 不显著 | 可能存在 [[Multicollinearity]] |
| F 不显著、$R^2$ 很低 | 模型整体解释力弱 |
| 加变量后 F 不通过 | 新变量组贡献有限 |
| 经典 F 与稳健 F 差异大 | 标准误假设可能不稳健 |

## 检查点

- 确认限制个数和自由度。
- 确认模型是否嵌套。
- 有异方差/自相关时使用稳健 Wald/F 检验。

## 来自课程位置

- [[03_多元线性回归]]

## 关联卡片

- [[F-test]]
- [[t Test]]
- [[R-squared]]
- [[Variance Inflation Factor]]
