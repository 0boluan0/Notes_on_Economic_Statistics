---
aliases:
- Newey-West
- Newey-West Standard Errors
- HAC standard errors
- HAC标准误
tags:
- system
- econometrics
---
# Newey-West

## 诊断目标

Newey-West 提供异方差-自相关一致的协方差矩阵估计，让时间序列回归的推断在 HAC 情况下更稳健。

## 什么时候用

- 残差可能同时有 [[Heteroskedasticity]] 和 [[Autocorrelation]]。
- 你不想明确建模 AR 误差结构，只想修正标准误。
- 样本量足够大，做渐近推断。

## 核心结构

Newey-West 用残差和解释变量构造长期协方差矩阵，并对滞后协方差加权：

$$
S=\hat\Omega_0+\sum_{j=1}^L w_j(\hat\Omega_j+\hat\Omega_j')
$$

再代入 OLS sandwich 协方差。

## 检查点

- 带宽 $L$ 会影响结果，要做敏感性检查。
- 纯异方差横截面数据通常用 [[White Robust Standard Errors]] 即可。
- Newey-West 修正标准误，不改变 OLS 系数。

## 常见错误

- 把 Newey-West 当成解决内生性的工具。
- 小样本中盲目使用很大带宽。
- 有明显动态设定遗漏时，只修标准误不改模型。

## 来自课程位置

- [[08_自相关]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[Autocorrelation Diagnosis]]
- [[White Robust Standard Errors]]
- [[Cochrane-Orcutt]]
