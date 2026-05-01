---
aliases:
- Condition Index
- CI
- 条件指数
tags:
- system
- econometrics
---
# Condition Index

## 诊断目标

Condition Index 从整个设计矩阵的特征值角度诊断多重共线性和矩阵病态。

## 公式

对标准化后的 $X'X$ 或相关矩阵取特征值 $\lambda_k$：

$$
CI_k=\sqrt{\frac{\lambda_{\max}}{\lambda_k}}
$$

## 判断

| CI | 解释 |
| --- | --- |
| < 10 | 通常无严重问题 |
| 10-30 | 中度共线性 |
| >= 30 | 严重共线性信号 |

Belsley 判据还要求看方差分解比例：若高 CI 维度上多个变量方差比例都很高，共线性更可信。

## 使用边界

- VIF 看单个变量；Condition Index 看整体矩阵。
- 变量尺度会影响诊断，通常先中心化/标准化。
- 高 CI 后仍要回到变量组合解释，不要只看数字。

## 来自课程位置

- [[06_多重共线性]]

## 关联卡片

- [[Multicollinearity]]
- [[Variance Inflation Factor]]
- [[Eigenvalues]]
