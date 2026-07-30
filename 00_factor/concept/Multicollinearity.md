---
aliases:
- Multicollinearity
- 多重共线性
tags:
  - concept
  - econometrics
---
# Multicollinearity

## 先记一句话

多重共线性是解释变量之间存在强线性关系，导致单个系数很难被精确区分。

## 它是什么

完全多重共线性表示某个解释变量能被其他解释变量线性表示，导致 $X'X$ 不可逆。高度但不完全的共线性会让标准误变大、系数不稳定。

## 解决什么判断

它回答：“回归里多个解释变量是否在传递几乎相同的信息？”

## 最小例子

同时放入“年龄”和“出生年份”，在同一年样本中二者几乎完全线性相关，可能导致共线性问题。

## 易混点

- 多重共线性本身不必然导致 OLS 有偏。
- 它主要影响精度和解释稳定性，而不是因果识别的核心外生性。
- 诊断指标见 [[Variance Inflation Factor]] 和 [[Condition Index]]；是否处理要结合研究目标。

## 来自课程位置

- [[06_多重共线性]]

## 关联卡片

- [[Variance Inflation Factor]]
- [[Condition Index]]
- [[Ridge Regression]]
- [[PCA]]
