---
aliases: [Geometric Series, Geometric Sum, 几何级数, 等比数列求和]
tags: [concept, discrete-mathematics, series]
type: concept
---
# Geometric Series

有限几何和满足

$$
\sum_{i=0}^{n}r^i=\frac{1-r^{n+1}}{1-r}\quad(r\ne1),
$$

而 $r=1$ 时和为 $n+1$。无限几何级数 $\sum_{i=0}^{\infty}r^i$ 当且仅当 $|r|<1$ 时收敛，此时和为 $1/(1-r)$。

收敛条件不能省略；闭式来自有限部分和的极限，而不是把“无穷项”直接当普通代数对象操作。

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
