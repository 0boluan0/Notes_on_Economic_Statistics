---
aliases:
- Outlier Detection
- 异常值检测
- 离群值检测
- influential observation diagnostics
tags:
- system
- econometrics
---
# Outlier Detection

## 诊断目标

识别残差异常、高杠杆和对回归结果有强影响的观测，并决定如何透明处理。

## 三类点

| 类型 | 判断 |
| --- | --- |
| 垂直异常值 | 残差大，但解释变量不极端 |
| 高杠杆点 | 解释变量取值极端 |
| 影响点 | 删除后系数明显变化，常用 [[Cook's Distance]] 判断 |

## 常用指标

- 学生化残差：$|r_i|>2$ 或 $|r_i|>3$ 需要检查。
- 杠杆值：$h_{ii}=x_i'(X'X)^{-1}x_i$。
- [[Cook's Distance]]：同时考虑残差和杠杆。
- DFFITS：看单个点对预测值的影响。

## 处理流程

1. 先确认是否数据录入或单位错误。
2. 若是真实极端值，保留并做稳健性分析。
3. 报告包含和排除该点的结果差异。
4. 若异常来自模型设定错误，优先修模型。
5. 可考虑 [[Robust Regression]]。

## 常见错误

- 因为点“难看”就删除。
- 只报告删除后的显著结果。
- 把异常值问题和 [[Heteroskedasticity]]、[[Model Misspecification]] 混在一起不诊断。

## 来自课程位置

- [[04_模型设定]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[Cook's Distance]]
- [[Residual]]
- [[Robust Regression]]
- [[Model Validation]]
