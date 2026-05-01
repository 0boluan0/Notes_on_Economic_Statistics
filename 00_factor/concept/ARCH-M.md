---
aliases:
- ARCH-M
- ARCH-in-Mean
- ARCHM
- ARCH in Mean
tags:
- concept
- 时间序列
- 波动建模
---
# ARCH-M

## 先记一句话

ARCH-M 就是：**把条件方差放进均值方程，用来表达风险会影响期望收益**。

这里的 M 是 in-Mean。

## 它是什么

基本形式：
$$
y_t=\mu_t+\varepsilon_t,
$$
$$
\mu_t=\beta+\delta h_t,
$$
$$
h_t=\alpha_0+\sum_{i=1}^{q}\alpha_i\varepsilon_{t-i}^2.
$$

若 $\delta>0$，表示波动越高，期望收益越高。

## 它解决什么判断

ARCH-M 用来检验风险补偿：

> 条件风险是否进入资产收益的条件均值？

普通 ARCH/GARCH 只让方差变；ARCH-M 让方差反过来影响均值。

## 常见误区

- ARCH-M 不是另一种方差方程，而是在均值方程里加入 $h_t$。
- $\delta$ 的经济解释要结合变量是收益、利差还是宏观量。
- 先确认方差模型合理，再解释均值中的风险项。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#3.2 ARCH-M|时间序列 04：ARCH-M]]

## 关联卡片

- [[ARCH]]
- [[GARCH]]
- [[Conditional Heteroskedasticity]]
- [[Volatility Clustering]]

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
