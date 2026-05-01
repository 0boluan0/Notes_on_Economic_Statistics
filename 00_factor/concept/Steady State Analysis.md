---
aliases:
- Steady State Analysis
- Steady State
- 稳态分析
- 稳定状态
tags:
- concept
- economics
---
# Steady State Analysis

## 一句话记忆

稳态是系统的关键变量不再变化，或以固定速度同步增长的长期状态。

## 它是什么

Steady State Analysis 用来找长期均衡位置。在增长模型中，常见形式是人均资本不再变化：

$$
\dot{k}=0
$$

在标准 [[Solow Model]] 中：

$$
sf(k^*)=(n+\delta)k^*
$$

## 解决什么判断

- 经济长期会收敛到哪里。
- 当前资本低于或高于稳态时，资本会增加还是减少。
- 政策改变的是稳态水平，还是长期增长率。

## 最小例子

若 $sf(k)>(n+\delta)k$，实际投资超过维持人均资本不变所需投资，$k$ 上升；若 $sf(k)<(n+\delta)k$，$k$ 下降。

## 易混点

- 稳态不等于“所有总量都不增长”；在人口增长时，总资本和总产出仍可增长。
- 无技术进步的 Solow 模型中，人均变量在稳态不增长。
- 储蓄率提高通常提高稳态水平，但不改变长期人均增长率。

## 来自课程位置

- [[06_经济增长理论#3.3 模型的稳态分析]]

## 关联卡片

- [[Solow Model]]
- [[Solow Steady State Calculation]]
- [[Solow Model Interpretation]]
- [[Economic Growth]]
- [[Growth Theory-hub]]

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
