---
aliases:
- Multivariate Mean Inference Map
- 多元均值推断选择图
tags:
- framework
- multivariate statistics
---
# Multivariate Mean Inference Map

## 什么时候用

当问题的对象是一个均值向量，而不是单个均值时，用这张图判断检验路线。

## 如何识别题型

| 题型 | 识别信号 | 首选工具 |
|---|---|---|
| 单样本均值向量 vs 目标 | $H_0:\mu=\mu_0$ | [[Hotelling T2 Test Procedure]] |
| 配对样本 | 同一对象两次测量 | [[Paired Mean Vector Comparison]] |
| 两独立样本，协方差相等 | 两组不同对象 | [[Two-Sample Hotelling T2 Test]] |
| 多组均值向量 | $g\geq3$ 个总体 | [[One-way MANOVA Procedure]] |
| 多个分量区间 | 同时报告多个均值分量 | [[Simultaneous Confidence Intervals]] 或 [[Bonferroni Method]] |

## 边界条件

- $S$ 或 $S_p$ 要可逆。
- 小样本精确推断依赖多元正态假设。
- 当 $n$ 接近或小于 $p$ 时，标准公式容易失效。

## 失败模式

- 把多个单变量 $t$ 检验当成联合检验。
- 配对数据被误处理成独立两样本。
- 协方差不等时仍使用 pooled covariance。

## 来自课程位置

- [[05_ 总体平均向量的推论]]
- [[06_比较多个均值向量comparisons of multivariate mean vectors]]

## 关联卡片

- [[Hotelling T2 Test]]
- [[MANOVA]]
- [[Confidence Ellipsoid]]
