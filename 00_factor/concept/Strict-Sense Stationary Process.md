---
aliases:
- Strict-Sense Stationary Process
- Strictly Stationary Process
- strict stationarity
- 严平稳过程
- 严平稳
tags:
- concept
- 时间序列
- 随机过程
---
# Strict-Sense Stationary Process

## 先记一句话

严平稳就是：**不管把整条过程往前或往后平移多少期，所有联合分布都不变**。

它比常用的弱平稳更强，因为它管的是整个分布，而不只是均值和方差。

## 它是什么

对任意时间点
$$
t_1,\dots,t_k
$$
和任意平移 $h$，如果
$$
(y_{t_1},\dots,y_{t_k})
$$
与
$$
(y_{t_1+h},\dots,y_{t_k+h})
$$
具有相同联合分布，则过程严平稳。

## 它解决什么判断

严平稳回答：

> 这个随机过程的完整概率结构是否不依赖日历时间？

它是理论上最强的平稳性定义，但实务建模里常用弱平稳，因为弱平稳只需要检查二阶矩。

## 和 [[Wide-Sense Stationary Process]] 的关系

如果二阶矩存在：
$$
\text{Strict stationarity}\Rightarrow \text{Wide-sense stationarity}.
$$

但宽平稳一般不能推出严平稳。

特殊情形：Gaussian process 的分布完全由均值和协方差决定，因此宽平稳可以推出严平稳。

## 常见误区

- 严平稳不是“均值不变”这么简单，而是所有有限维联合分布不变。
- 严平稳不自动保证方差存在；说它推出弱平稳时要加上二阶矩存在。
- 课程中建 ARMA 通常检查弱平稳，不是直接验证严平稳。

## 来自课程位置

- [[03_平稳时间序列模型#2.1 平稳性定义|时间序列 03：严平稳与弱平稳]]

## 关联卡片

- [[Stationarity]]
- [[Wide-Sense Stationary Process]]
- [[Ergodicity]]

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
