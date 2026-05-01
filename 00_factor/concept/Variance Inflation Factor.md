---
aliases:
- Variance Inflation Factor
- VIF
- 方差膨胀因子
tags:
- concept
- econometrics
---
# Variance Inflation Factor

## 先记一句话

VIF 衡量某个解释变量因被其他解释变量线性解释而导致系数方差膨胀多少倍。

## 它是什么

把第 $j$ 个解释变量对其他解释变量回归，得到 $R_j^2$：

$$
VIF_j=\frac{1}{1-R_j^2}
$$

## 解决什么判断

它回答：“这个变量的系数标准误是否因为共线性被放大了？”

## 最小例子

若 $R_j^2=0.9$，则 $VIF_j=10$，表示该系数方差约被放大 10 倍。

## 易混点

- VIF 是单个变量层面的诊断；整体矩阵病态看 [[Condition Index]]。
- VIF 高不等于必须删除变量，理论关键变量可以保留。
- VIF 不诊断内生性。

## 来自课程位置

- [[06_多重共线性]]

## 关联卡片

- [[Multicollinearity]]
- [[Condition Index]]
- [[Ridge Regression]]
