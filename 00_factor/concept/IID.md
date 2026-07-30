---
aliases:
- IID
- i.i.d.
- independent and identically distributed
- 独立同分布
- 独立同分布（i.i.d.）
tags:
  - concept
  - 概率论
  - 统计学
---
# IID

## 先记一句话

IID 就是：**一组随机变量相互独立，并且具有同一个边际分布**。

它比“不相关”强很多。

## 它是什么

随机变量序列 $(X_i)_{i\ge1}$ 是 i.i.d.，需要同时满足：

- independent：任意有限子族相互独立；
- identically distributed：每个 $X_i$ 具有同一个边际分布。

在时间序列应用中，如果再有
$$
E(X_t)=0,
$$
那么它自然满足鞅差条件。

## 它解决什么判断

IID 是许多概率极限定理和简单统计推断的标准假设，但“独立”与“同分布”是两个必须分别检查的条件。

在时间序列里，它常用来作为创新项的理想化假设，但金融收益或宏观数据经常达不到这么强。

## 和 MDS / White Noise 的关系

在零均值、二阶矩存在时：
$$
IID \Rightarrow [[Martingale Difference Sequence]] \Rightarrow [[White Noise]].
$$

IID 最强，因为它要求完整分布独立；white noise 最弱，只要求二阶不相关。

## 常见误区

- 不相关不等于独立。
- 同分布不等于独立。
- 有 ARCH 效应的序列可能是 MDS，但通常不是 IID，因为条件方差随过去变化。
- 两两独立不自动等于相互独立，见 [[Mutual Independence]]。

## 离散数学中的最小例子

重复独立抛掷同一枚硬币，令 $X_i$ 表示第 $i$ 次是否为正面，则 $X_i$ 是相互独立且同为 Bernoulli$(p)$ 分布，因此是 IID。

## 来自课程位置

- [[03_平稳时间序列模型#1.2 三种‘没有关系’的辨析|时间序列 03：IID/MDS/白噪声辨析]]

## 关联卡片

- [[Martingale Difference Sequence]]
- [[White Noise]]
- [[Independence vs. Uncorrelated]]
- [[00_factor/concept/ARCH|ARCH]]


## 最小例子

把 **IID** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。

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
