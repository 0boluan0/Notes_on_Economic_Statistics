---
aliases:
- Martingale Difference Sequence
- MDS
- martingale difference
- 鞅差序列
tags:
  - concept
  - 时间序列
  - 概率论
---
# Martingale Difference Sequence

## 先记一句话

鞅差序列就是：**给定过去信息后，当前扰动的条件期望为 0**。

公式是
$$
E(\varepsilon_t\mid\mathcal{F}_{t-1})=0.
$$

## 它是什么

$\mathcal{F}_{t-1}$ 表示 $t-1$ 期以前所有已知信息。

MDS 的意思是：即使你知道过去所有信息，也不能系统性预测当前扰动的方向。

它非常适合表达“均值不可预测”，比如有效市场假说下的超额收益。

## 一个最小例子

ARCH 模型中：
$$
\varepsilon_t=\nu_t\sqrt{h_t},
$$
其中 $h_t$ 由过去信息决定，而
$$
E(\nu_t)=0.
$$

所以
$$
E(\varepsilon_t\mid\mathcal{F}_{t-1})=0.
$$

它可以是 MDS，但因为条件方差 $h_t$ 随时间变化，所以通常不是 IID。

## 和 White Noise 的关系

若二阶矩存在，MDS 蕴含不相关，因此蕴含 [[White Noise]] 的核心二阶性质。

但 white noise 不一定是 MDS，因为不相关不代表条件期望为 0。

## 常见误区

- MDS 管的是条件均值，不是条件方差。
- MDS 可以有条件异方差。
- MDS 不等于独立；它只是说明过去不能预测当前均值。

## 来自课程位置

- [[03_平稳时间序列模型#1.2 三种‘没有关系’的辨析|时间序列 03：IID/MDS/白噪声辨析]]
- [[04_波动建模 Modeling Volatility#2.2 ARCH|时间序列 04：ARCH 可以是 MDS]]

## 关联卡片

- [[IID]]
- [[White Noise]]
- [[ARCH]]
- [[Conditional Heteroskedasticity]]
- [[Efficient Market Hypothesis]]


## 最小例子

把 **Martingale Difference Sequence** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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
