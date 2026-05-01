---
aliases:
- Jarque-Bera Test
- JB Test
- JB检验
- 正态性检验
tags:
- system
- econometrics
- statistics
---
# Jarque-Bera Test

## 诊断目标

检验残差或样本分布是否符合正态分布的偏度和峰度特征。

## 统计量

$$
JB=\frac{n}{6}\left(S^2+\frac{(K-3)^2}{4}\right)
$$

其中 $S$ 是偏度，$K$ 是峰度。

## 怎么读

原假设：分布为正态。

- p 值小：拒绝正态性。
- p 值大：没有足够证据拒绝正态性。

## 易混点

- 正态性不是 OLS 无偏或一致的必要条件。
- 小样本精确 t/F 推断更依赖正态性。
- 大样本下常靠渐近正态和稳健标准误。

## 来自课程位置

- [[03_多元线性回归]]

## 关联卡片

- [[OLS Basics]]
- [[Asymptotic Theory]]
- [[t Test]]
