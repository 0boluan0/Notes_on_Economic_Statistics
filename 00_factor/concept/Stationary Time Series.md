---
aliases:
- Stationary Time Series
- 平稳时间序列
tags:
- concept
---
# Stationary Time Series

## 它是什么
- 「Stationary Time Series」是指均值、方差恒定且自协方差仅与滞后有关的时间序列。

## 最小可检索信息
- 定义：均值、方差恒定且自协方差仅与滞后有关的时间序列。
- 符号/公式：$E[x_t]=\mu，\mathrm{Var}(x_t)=\sigma^2。$
- 最小例子：白噪声序列。

## 关联卡片
- [[ARMA Model Identification Steps]]
- [[Box-Jenkins Method]]

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
