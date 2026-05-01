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

IID 就是：**每一期随机变量彼此独立，并且来自同一个分布**。

它比“不相关”强很多。

## 它是什么

序列 $X_t$ 是 i.i.d.，需要同时满足：

- independent：不同期之间相互独立；
- identically distributed：每一期边际分布相同。

如果再有
$$
E(X_t)=0,
$$
那么它自然满足鞅差条件。

## 它解决什么判断

IID 是很多概率极限定理和简单统计推断的强假设。

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

## 来自课程位置

- [[03_平稳时间序列模型#1.2 三种‘没有关系’的辨析|时间序列 03：IID/MDS/白噪声辨析]]

## 关联卡片

- [[Martingale Difference Sequence]]
- [[White Noise]]
- [[Independence vs. Uncorrelated]]
- [[ARCH]]

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
