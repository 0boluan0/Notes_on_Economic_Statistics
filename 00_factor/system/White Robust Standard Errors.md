---
aliases:
- White Robust Standard Errors
- White Robust SE
- HC standard errors
- HC标准误
- White稳健标准误
- 稳健标准误
tags:
- system
- econometrics
---
# White Robust Standard Errors

## 诊断目标

在存在未知形式异方差时，不改变 OLS 系数，只把协方差矩阵估计改成异方差一致形式。

## 核心公式

HC0 形式：

$$
\widehat{Var}_{HC0}(\hat\beta)
=(X'X)^{-1}\left(\sum_{i=1}^n x_ix_i'\hat u_i^2\right)(X'X)^{-1}
$$

常用软件会使用 HC1、HC2、HC3 等小样本或杠杆值修正。

## 什么时候用

- 横截面数据存在异方差。
- 不知道可靠的方差权重。
- 主要目标是让 t/F 推断更可信。

## 不解决什么

- 不修正系数估计本身。
- 不处理 [[Endogeneity]]。
- 不处理时间序列自相关；这时看 [[Newey-West]]。
- 不让 OLS 重新变成最有效估计。

## 报告方式

报告回归系数时说明使用 heteroskedasticity-robust standard errors。若结论依赖标准误选择，应把经典标准误与稳健标准误差异说清楚。

## 来自课程位置

- [[07_异方差]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[Heteroskedasticity]]
- [[Heteroscedasticity Diagnosis]]
- [[White Test]]
- [[Newey-West]]
