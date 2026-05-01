---
aliases:
- AR(1) stationarity
- AR(1) 平稳性条件
- AR(1)过程的平稳性
tags:
- proof
- 时间序列
---
# AR(1) stationarity

## 假设

考虑 AR(1)：
$$
y_t=a_0+a_1y_{t-1}+\varepsilon_t,
$$
其中 $\varepsilon_t$ 是白噪声：
$$
E(\varepsilon_t)=0,\qquad \operatorname{Var}(\varepsilon_t)=\sigma^2.
$$

目标是证明：

> AR(1) 协方差平稳的核心条件是 $|a_1|<1$。

## 推导链

不断向前迭代：
$$
y_t=a_0\sum_{i=0}^{t-1}a_1^i+a_1^t y_0+\sum_{i=0}^{t-1}a_1^i\varepsilon_{t-i}.
$$

若
$$
|a_1|<1,
$$
则
$$
a_1^t y_0\to0
$$
且几何级数收敛。过程可以写成从无限过去开始的稳定表示：
$$
y_t=\frac{a_0}{1-a_1}+\sum_{i=0}^{\infty}a_1^i\varepsilon_{t-i}.
$$

## 均值

取期望：
$$
E(y_t)=\frac{a_0}{1-a_1}.
$$

这是常数，不依赖 $t$。

## 方差

因为白噪声不同期不相关：
$$
\operatorname{Var}(y_t)
=\sigma^2\sum_{i=0}^{\infty}a_1^{2i}
=\frac{\sigma^2}{1-a_1^2}.
$$

这要求 $|a_1|<1$ 才有限。

## 自协方差

对滞后 $s$：
$$
\gamma_s
=\operatorname{Cov}(y_t,y_{t-s})
=\frac{\sigma^2a_1^s}{1-a_1^2}.
$$

所以
$$
\rho_s=\frac{\gamma_s}{\gamma_0}=a_1^s.
$$

自协方差只依赖滞后 $s$，不依赖具体时间 $t$。

## 结论

当 $|a_1|<1$ 时，AR(1) 满足弱平稳的三个条件：

- 均值常数；
- 方差有限且常数；
- 自协方差只依赖滞后。

若 $|a_1|\geq1$，初始条件或冲击影响不会衰减，方差或均值结构无法稳定。

## 来自课程位置

- [[03_平稳时间序列模型#2.2.1 AR(1)的平稳性判断|时间序列 03：AR(1) 平稳性推导]]

## 关联卡片

- [[Autoregressive Model]]
- [[Stationarity]]
- [[ARMA]]
- [[Yule-Walker equations]]
- [[Random Walk]]

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
