---
aliases:
- Classification Rule Selection
- 判别分类规则选择
tags:
- framework
- multivariate statistics
---
# Classification Rule Selection

## 什么时候用

当题目要求把新观测分到已知类别，并给出先验概率、误分类成本或密度函数时，用这张卡选规则。

## 如何识别

| 题目给什么 | 用什么 |
|---|---|
| 先验概率、成本、密度 | [[Expected Cost of Misclassification]] 规则 |
| 两组协方差近似相等 | [[Fisher Linear Discriminant]] |
| 要估计分类表现 | [[Actual Error Rate]] 或 [[Jackknife Classification]] |

## 为什么这样看

分类不是只看“更像哪一类”，还要看先验概率和误判成本。成本差异大时，阈值必须偏向低风险选择。

## 边界条件

- Fisher 线性判别依赖 pooled covariance 的合理性。
- 密度比规则要求能估计或假设各总体密度。

## 失败模式

- 把 ECM 误连到计量经济学的 Error Correction Model。
- 成本不等时仍按 $f_1(x)>f_2(x)$ 分类。

## 来自课程位置

- [[11_分类与判别Discrimination and Classifications]]

## 关联卡片

- [[Expected Cost of Misclassification]]
- [[Fisher Discriminant Procedure]]
- [[Jackknife Classification]]
