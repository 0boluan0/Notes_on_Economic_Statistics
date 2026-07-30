---
aliases: [Stars and Bars, Balls and Dividers, 隔板法]
tags: [concept, discrete-mathematics, counting]
---
# Stars and Bars

非负整数方程
$$
x_1+\cdots+x_k=n
$$
的解数为
$$
\binom{n+k-1}{k-1},
$$
因为每个解对应 $n$ 个相同 stars 与 $k-1$ 个 bars 的排列。

若要求 $x_i\ge1$，先令 $y_i=x_i-1$，解数为 $\binom{n-1}{k-1}$（需 $n\ge k$）。存在上界时不能直接套公式，通常需要容斥。

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
