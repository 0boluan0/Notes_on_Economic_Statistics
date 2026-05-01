---
aliases:
- Random Walk
- random walk
- 随机游走
tags:
- concept
- 时间序列
---
# Random Walk

## 先记一句话

随机游走就是：**今天的水平等于昨天的水平加上一个新冲击**。

最基本形式：
$$
y_t=y_{t-1}+\varepsilon_t.
$$

## 它是什么

随机游走的每个冲击都会永久进入水平值。

展开可得：
$$
y_t=y_0+\sum_{i=1}^{t}\varepsilon_i.
$$

所以它不是围绕固定均值波动，而是不断累积冲击。

## 它解决什么判断

随机游走是单位根过程的核心例子。

题目里如果出现：

- unit root；
- I(1)；
- first difference stationary；
- no mean reversion；
- shocks have permanent effects；

就应该想到 random walk。

## 一个最小例子

若
$$
y_t=y_{t-1}+\varepsilon_t,\qquad \varepsilon_t\sim WN(0,\sigma^2),
$$
则
$$
\Delta y_t=\varepsilon_t.
$$

原序列非平稳，但一阶差分平稳。

## 常见误区

- 随机游走不是“完全随机没有结构”；它有很强的持久性结构。
- 随机游走的变化量可能是白噪声，但水平值不是白噪声。
- 两个独立随机游走做水平回归容易产生 [[Spurious Regression]]。

## 来自课程位置

- [[03_平稳时间序列模型#1.1.3 ARIMA过程|时间序列 03：单位根与 ARIMA]]
- [[07_协整和误差修正模型#2.1 协整的定义|时间序列 07：I(1) 与协整]]

## 关联卡片

- [[Unit Root Test]]
- [[First Difference]]
- [[ARIMA]]
- [[Stationarity]]
- [[Spurious Regression]]
- [[Cointegration]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
