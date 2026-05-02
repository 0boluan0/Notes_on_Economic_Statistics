---
aliases:
- Simultaneous Confidence Intervals
- Simultaneous Confidence Interval
- 同时置信区间
tags:
- concept
- multivariate statistics
---
# Simultaneous Confidence Intervals

>[!note] 一句话记忆
> 同时置信区间是在控制整体置信水平的前提下，给多个均值分量或线性组合同时构造区间。

## 它是什么

对均值分量 $\mu_i$，Hotelling 型同时区间可写成
$$
\bar X_i\pm
\sqrt{\frac{p(n-1)}{n-p}F_{p,n-p}(\alpha)}
\sqrt{\frac{s_{ii}}{n}}.
$$

## 解决什么判断

- 多个变量的均值分量能否同时被区间覆盖。
- 多重比较时如何避免置信水平被稀释。
- 什么时候用 Hotelling 区间，什么时候用 Bonferroni 区间。

## 最小例子

同时报告 BOD 和 SS 两个均值差的置信区间，需要控制两个区间一起成立的置信水平。

## 易混点

- 同时区间通常比单独区间宽。
- Bonferroni 区间和 Hotelling 区间控制目标相似，但临界值来源不同。

## 来自课程位置

- [[05_ 总体平均向量的推论#1.5. 置信区域与同时置信区间|第5章 3 置信区域与同时置信区间]]

## 关联卡片

- [[Confidence Ellipsoid]]
- [[Bonferroni Method]]
- [[Hotelling T2 Test]]
