---
aliases:
- Modified Duration
- 修正久期
tags:
  - concept
  - fixed-income
---
# Modified Duration

## 先记一句话

Modified Duration 把 Macaulay Duration 转成“收益率变 1 个单位时，价格百分比大约变多少”。

## 它是什么

离散复利下：

$$
D_{mod}=\frac{D_M}{1+y/m}
$$

价格变化的一阶近似为：

$$
\frac{\Delta P}{P}\approx -D_{mod}\Delta y
$$

## 解决什么判断

它回答：“利率小幅上升或下降时，债券价格大约按百分比变化多少？”

## 最小例子

修正久期为 5，收益率上升 0.01，则价格约下降 $5\%$。

## 易混点

- 修正久期是百分比敏感度；金额敏感度见 [[Dollar Duration]] 与 [[Basis Point Value (BPV)]]。
- 连续复利下，Macaulay Duration 和 Modified Duration 在形式上可相等。
- 利率大幅变化时要加 [[Convexity]] 修正。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Macaulay Duration]]
- [[Modified Duration Calculation]]
- [[Dollar Duration]]
- [[Convexity]]
