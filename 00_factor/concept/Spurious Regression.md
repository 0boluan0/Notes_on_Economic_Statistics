---
aliases:
- Spurious Regression
- spurious regression
- 伪回归
- 虚假回归
tags:
- concept
- 时间序列
- 计量经济学
---
# Spurious Regression

## 先记一句话

伪回归就是：**两个没有真实关系的非平稳序列，因为各自带趋势或随机趋势，回归出来却显得很显著**。

它是时间序列里最危险的假象之一。

## 它是什么

两个独立随机游走：
$$
y_t=y_{t-1}+u_t,
\qquad
x_t=x_{t-1}+v_t.
$$

即使 $u_t$ 和 $v_t$ 完全无关，回归
$$
y_t=\alpha+\beta x_t+e_t
$$
也可能得到高 $R^2$ 和显著 t 统计量。

问题不在 OLS 算错，而在非平稳序列破坏了普通推断条件。

## 它解决什么判断

当水平回归结果很好，但变量可能是 $I(1)$ 时，要问：

> 这是长期均衡关系，还是伪回归？

判断入口是：

- 先做 [[Unit Root Test]]；
- 如果变量都是 $I(1)$，再检验 [[Cointegration]]；
- 如果没有协整，不要信水平回归的显著性。

## 一个最小识别法

如果水平回归残差是平稳的，可能存在协整。

如果残差仍然有单位根，很可能是伪回归。

这就是 [[Engle-Granger Two-Step Test]] 的核心逻辑。

## 常见误区

- 高 $R^2$ 不代表长期关系真实。
- 显著 t 值不代表可以忽略单位根。
- 差分可以避免伪回归，但可能丢掉协整中的长期信息。

## 来自课程位置

- [[07_协整和误差修正模型#2.1 协整的定义|时间序列 07：协整与伪回归背景]]
- [[07_协整和误差修正模型#3.1 EG两步法|时间序列 07：用残差平稳性区分协整和伪回归]]

## 关联卡片

- [[Random Walk]]
- [[Unit Root Test]]
- [[Cointegration]]
- [[Engle-Granger Two-Step Test]]
- [[First Difference]]

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
