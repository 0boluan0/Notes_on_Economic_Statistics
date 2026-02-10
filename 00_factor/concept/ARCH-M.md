---
aliases:
- ARCH-in-Mean
- ARCHM
- ARCH-M
- ARCH
tags:
- 时间序列
- 波动建模
- concept
---
ARCH-M（ARCH-in-Mean）在均值方程中引入条件方差（或其函数），刻画“风险—收益”正相关：波动越大，期望收益越高。

## 形式

$$
\begin{cases}
y_t = \mu_t + \varepsilon_t, \\
\mu_t = \beta + \delta h_t,\quad (\delta>0), \\
h_t = \alpha_0 + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2
\end{cases}
$$

要点：
- $\delta>0$ 表示均值受波动正向影响；
- 可与 GARCH 结合（在 $h_t$ 使用 GARCH 结构）。

## Connections

- 相关：[[ARCH]]、[[GARCH]]、[[TARCH]]、[[EGARCH]]
- 估计：[[Maximum Likelihood Estimation|极大似然估计]]

## $source_notes$

- [[04_波动建模 Modeling Volatility#3.2 ARCH-M]]

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
