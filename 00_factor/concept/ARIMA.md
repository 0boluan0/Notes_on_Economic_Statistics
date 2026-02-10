---
aliases:
- 自回归积分移动平均模型
- ARIMA模型
- ARIMA
tags:
- 时间序列
- concept
---
ARIMA 模型（Autoregressive Integrated Moving Average）在 ARMA 基础上引入差分阶数 $d$，用于描述经差分后平稳的非平稳序列。

## 形式

对原序列 $y_t$ 做 $d$ 阶差分 $\Delta^d y_t$，若 $\Delta^d y_t$ 可由 $\text{ARMA}(p,q)$ 刻画，则称 $y_t \sim \text{ARIMA}(p,d,q)$。

## 何时使用

- 特征根含 1（单位根）导致非平稳，需要差分以达成平稳再建模。

## Connections

- 相关：[[ARMA]]、[[Unit Root Test|单位根检验]]、[[Stationarity|平稳性]]

## source_notes

- [[03_平稳时间序列模型#1.1.3 ARIMA过程]]

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
