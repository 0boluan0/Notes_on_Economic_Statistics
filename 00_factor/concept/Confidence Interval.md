---
aliases:
- Confidence Interval
- CI
- 置信区间
tags:
- concept
- statistics
- econometrics
---
# Confidence Interval

## 先记一句话

置信区间用“估计值 ± 临界值 × 标准误”表达参数估计的不确定性。

## 它是什么

对回归系数 $\beta_j$，常见置信区间为：

$$
\hat\beta_j\pm c\cdot se(\hat\beta_j)
$$

其中 $c$ 是对应置信水平下的临界值。

## 解决什么判断

它回答：“参数真值在多大范围内比较合理？”

## 最小例子

若 $\hat\beta_1=0.8$，$se=0.1$，95% 近似置信区间为：

$$
0.8\pm1.96\times0.1=[0.604,0.996]
$$

## 易混点

- 置信区间不是“真值有 95% 概率落在这个已算出的区间里”的贝叶斯说法。
- 区间是否可靠依赖标准误是否可靠。
- 若区间不含 0，通常对应双侧显著性检验拒绝 $H_0:\beta_j=0$。

## 来自课程位置

- [[02_一元线性回归]]

## 关联卡片

- [[P-value]]
- [[t Test]]
- [[White Robust Standard Errors]]
