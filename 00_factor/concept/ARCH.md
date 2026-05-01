---
aliases:
- ARCH
- ARCH Model
- Autoregressive Conditional Heteroskedasticity
- 自回归条件异方差
tags:
- concept
- 时间序列
- 波动建模
---
# ARCH

## 先记一句话

ARCH 就是：**用过去冲击的平方解释当前条件方差**。

它建模的不是均值，而是波动率。

## 它是什么

ARCH(1)：
$$
\varepsilon_t=\nu_t\sqrt{h_t},
\qquad
h_t=\alpha_0+\alpha_1\varepsilon_{t-1}^2.
$$

其中：

- $\nu_t$ 通常是 i.i.d.、均值 0、方差 1；
- $h_t$ 是条件方差；
- $\alpha_0>0$ 保证基础方差；
- $\alpha_1\geq0$ 表示上一期冲击平方对本期波动的影响。

## 它解决什么判断

ARCH 用来刻画：

> 大残差之后，下一期方差更大。

如果均值模型残差没有自相关，但残差平方有自相关，就该考虑 ARCH/GARCH。

## 一个最小直觉

上一期市场大跌或大涨，$\varepsilon_{t-1}^2$ 都很大。

ARCH 不关心方向，只关心冲击大小。

所以正负大冲击都会提高下一期波动。

## 和 GARCH 的关系

ARCH 只用过去冲击平方解释方差。

[[GARCH]] 还加入过去的条件方差：
$$
h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1}.
$$

因此 GARCH 通常能用更少参数刻画更持久的波动。

## 常见误区

- ARCH 的核心是条件方差，不是残差均值。
- $\alpha_0$ 不能随便删，否则长期方差结构会出问题。
- ARCH 可以是 MDS，但不是 IID，因为条件方差随过去变化。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#2.2 ARCH|时间序列 04：ARCH 模型]]

## 关联卡片

- [[Conditional Heteroskedasticity]]
- [[Volatility Clustering]]
- [[GARCH]]
- [[ARCH LM Test]]
- [[Martingale Difference Sequence]]

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
