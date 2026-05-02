---
aliases:
- PCA vs Factor Analysis
- PCA 与因子分析比较
tags:
- framework
- multivariate statistics
---
# PCA vs Factor Analysis

## 什么时候用

当题目同时出现降维、解释方差、潜在因子、公共度、特殊方差时，用这张卡区分 PCA 和因子分析。

## 如何识别

| 问题目标 | 方法 |
|---|---|
| 用少数正交方向保留最大总方差 | [[PCA]] |
| 用少数潜在因子解释协方差结构 | [[Factor Analysis]] |
| 只需要压缩变量 | PCA |
| 需要解释共同因子和特殊方差 | 因子分析 |

## 为什么这样看

PCA 是代数分解：找方差最大的正交线性组合。因子分析是统计模型：假设观测变量由公共因子和特殊因子组成。

## 边界条件

- 因子分析需要选择因子数，还要解释因子载荷。
- PCA 不区分公共方差和特殊方差。

## 失败模式

- 把 PCA 的前几个主成分直接命名为潜在因子。
- 用因子分析却不检查公共度和特殊方差。

## 来自课程位置

- [[08_主成分分析principal component]]
- [[09_因子分析Factor Analysis and Inference for Structured]]

## 关联卡片

- [[PCA Procedure]]
- [[Factor Analysis PC Method]]
- [[Factor Loadings]]
