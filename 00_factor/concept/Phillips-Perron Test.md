---
aliases:
- Phillips-Perron Test
- PP Test
- PP检验
- PP
- 菲利普斯-佩龙检验
tags:
- concept
- 时间序列
- 计量经济学
---
# Phillips-Perron Test

## 先记一句话

PP 检验就是：**不在回归里加很多滞后项，而是直接修正单位根检验统计量来处理自相关和异方差**。

它和 ADF 一样，通常检验：
$$
H_0:\text{存在单位根}.
$$

## 它是什么

PP 检验从 Dickey-Fuller 回归出发，但对统计量做非参数修正。

核心区别：

- [[Augmented Dickey-Fuller Test]]：参数化处理，自行加入滞后差分项；
- PP：非参数修正，调整统计量和标准误。

## 它解决什么判断

当你担心误差存在异方差或自相关，但又不想完全依赖 ADF 的滞后阶数选择时，可以用 PP 做稳健性对照。

## 常见误区

- PP 不是“比 ADF 永远更好”，它也依赖带宽/截断滞后选择。
- PP 和 ADF 的原假设方向一样，都是单位根。
- 如果 ADF 和 PP 结论冲突，要检查趋势项、结构突变、样本长度和滞后/带宽设定。

## 来自课程位置

- [[07_协整和误差修正模型#3.1 EG两步法|时间序列 07：单位根检验作为协整预检]]

## 关联卡片

- [[Unit Root Test]]
- [[Augmented Dickey-Fuller Test]]
- [[Stationarity Tests Comparison]]
- [[Newey-West]]

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
