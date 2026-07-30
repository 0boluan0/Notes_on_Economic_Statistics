---
aliases: [Expected Value, Expectation, 期望]
tags: [concept, discrete-mathematics, probability]
---
# Expected Value

离散随机变量 $X$ 的期望为
$$
\mathbb E[X]=\sum_x x\Pr(X=x),
$$
前提是相应级数有定义。它是长期加权平均，不必是 $X$ 能取到的值。

对函数 $g$，LOTUS 给出 $\mathbb E[g(X)]=\sum_xg(x)\Pr(X=x)$。多个变量之和可用 [[Linearity of Expectation]]。

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
