---
aliases:
- Conditional Heteroskedasticity
- conditional heteroskedasticity
- 条件异方差
tags:
- concept
- 时间序列
- 波动建模
---
# Conditional Heteroskedasticity

## 先记一句话

条件异方差就是：**给定过去信息后，当前误差的方差会随时间变化**。

公式上看：
$$
\operatorname{Var}(\varepsilon_t\mid\mathcal{F}_{t-1})=h_t.
$$

$h_t$ 不是常数，而是由过去冲击或过去方差决定。

## 它是什么

均值可能已经不可预测：
$$
E(\varepsilon_t\mid\mathcal{F}_{t-1})=0,
$$
但方差仍然可预测：
$$
E(\varepsilon_t^2\mid\mathcal{F}_{t-1})=h_t.
$$

这就是为什么序列可以是 [[Martingale Difference Sequence]]，但仍需要 [[ARCH]] / [[GARCH]] 建模。

## 一个最小例子

ARCH(1)：
$$
\varepsilon_t=\nu_t\sqrt{h_t},
\qquad
h_t=\alpha_0+\alpha_1\varepsilon_{t-1}^2.
$$

如果上一期冲击很大，$\varepsilon_{t-1}^2$ 很大，本期条件方差 $h_t$ 就会变大。

## 它解决什么判断

条件异方差回答：

> 残差均值已经像白噪声，但波动大小是否仍有可预测结构？

如果残差 ACF 干净，但残差平方 ACF 显著，就该怀疑条件异方差。

## 常见误区

- 条件异方差不是均值自相关；它是方差动态。
- 残差是白噪声不代表残差平方也没有结构。
- ARCH/GARCH 不是替代 ARMA；通常先建均值模型，再对残差波动建模。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#1.1 为什么要进行波动建模|时间序列 04：波动建模动机]]
- [[04_波动建模 Modeling Volatility#2. ARCH,GARCH|时间序列 04：ARCH/GARCH]]

## 关联卡片

- [[Volatility Clustering]]
- [[ARCH]]
- [[GARCH]]
- [[ARCH LM Test]]
- [[McLeod-Li Test]]
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
