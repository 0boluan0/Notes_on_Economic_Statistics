---
aliases:
- Stationarity
- Stationary Time Series
- stationary process
- 平稳性
- 平稳时间序列
tags:
- concept
- 时间序列
---
# Stationarity

## 先记一句话

平稳性就是：**时间序列的统计规律不随时间本身改变**。

你可以把它理解成：

> 这条序列虽然在波动，但它的均值、方差和相关结构没有一路漂走。

如果序列不平稳，很多 ARMA、ACF、普通回归推断都会变得不可靠。

## 它是什么

课程里最常用的是弱平稳，也叫 covariance stationarity。

它要求：

1. 均值恒定：
   $$
   E(y_t)=\mu.
   $$
2. 方差恒定：
   $$
   \operatorname{Var}(y_t)=\sigma^2.
   $$
3. 自协方差只依赖滞后，不依赖具体时间：
   $$
   \operatorname{Cov}(y_t,y_{t-s})=\gamma_s.
   $$

严平稳更强，要求任意有限维联合分布在时间平移后不变。

## 一个最小例子

白噪声
$$
\varepsilon_t\sim WN(0,\sigma^2)
$$
是平稳的：

- 均值一直是 0；
- 方差一直是 $\sigma^2$；
- 非零滞后协方差为 0。

随机游走
$$
y_t=y_{t-1}+\varepsilon_t
$$
通常不平稳，因为它的方差会随时间增长。

## 它在题里负责什么

- 判断能不能直接建 [[ARMA]]。
- 判断 ACF/PACF 是否有稳定意义。
- 决定是否需要 [[First Difference]] 或去趋势。
- 为 [[Cointegration]] 做前置判断：协整通常讨论 $I(1)$ 变量的线性组合是否 $I(0)$。

## 和严平稳、弱平稳的关系

- [[Strict-Sense Stationary Process]]：看整个分布是否随时间平移不变。
- [[Wide-Sense Stationary Process]]：只看均值、方差、自协方差这些二阶矩。

如果二阶矩存在：

$$
\text{严平稳} \Rightarrow \text{弱平稳}.
$$

反过来一般不成立。若过程是 Gaussian，弱平稳和严平稳才等价。

## 常见误区

- “看起来上下波动”不等于平稳；要看均值、方差和协方差结构。
- 趋势平稳和差分平稳不同：前者去趋势，后者差分。
- 非平稳变量之间回归可能产生 [[Spurious Regression]]。

## 来自课程位置

- [[03_平稳时间序列模型#2. 平稳性|时间序列 03：平稳性定义与例子]]
- [[03_平稳时间序列模型#2.2.1 AR(1)的平稳性判断|时间序列 03：AR(1) 平稳性]]
- [[07_协整和误差修正模型#2.1 协整的定义|时间序列 07：协整要求变量单整]]

## 关联卡片

- [[Strict-Sense Stationary Process]]
- [[Wide-Sense Stationary Process]]
- [[AR(1) stationarity]]
- [[Random Walk]]
- [[Unit Root Test]]
- [[ARMA]]
- [[First Difference]]
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
