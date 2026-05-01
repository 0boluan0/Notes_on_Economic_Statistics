---
aliases:
- Autoregressive Model
- AR Model
- AR
- 自回归模型
tags:
- concept
- 时间序列
---
# Autoregressive Model

## 先记一句话

AR 模型就是：**当前值主要由自己的过去值解释**。

AR$(p)$ 写作
$$
y_t=c+\phi_1y_{t-1}+\cdots+\phi_py_{t-p}+\varepsilon_t.
$$

## 它解决什么判断

AR 模型适合描述“过去水平会影响现在水平”的平稳序列。

如果 PACF 在 $p$ 阶后截尾，而 ACF 拖尾，就优先怀疑 AR$(p)$。

## 一个最小例子

AR(1)：
$$
y_t=a_0+a_1y_{t-1}+\varepsilon_t.
$$

若
$$
|a_1|<1,
$$
冲击影响会逐渐衰减，过程可以平稳。具体推导见 [[AR(1) stationarity]]。

## 常见误区

- AR 的 “R” 是 regressive on its own past，不是普通横截面回归。
- AR 阶数不是看 ACF 截尾，而是重点看 PACF 截尾。
- AR 只处理均值动态；残差方差动态要另看 [[ARCH]] / [[GARCH]]。

## 来自课程位置

- [[03_平稳时间序列模型#1.1.1 AR过程|时间序列 03：AR 过程]]
- [[03_平稳时间序列模型#2.2.1 AR(1)的平稳性判断|时间序列 03：AR(1) 平稳性]]

## 关联卡片

- [[ARMA]]
- [[AR(1) stationarity]]
- [[Partial Autocorrelation Function]]
- [[Autocorrelation Function]]
- [[Yule-Walker equations]]

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
