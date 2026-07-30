---
aliases: [Indicator Random Variable, Indicator Variable, 指示随机变量]
tags: [concept, discrete-mathematics, probability]
---
# Indicator Random Variable

事件 $A$ 的 indicator 为
$$
I_A(\omega)=\begin{cases}1,&\omega\in A,\\0,&\omega\notin A.\end{cases}
$$
因此
$$
\mathbb E[I_A]=\Pr(A).
$$

计数型随机变量常可写成多个 indicators 之和，再用 [[Linearity of Expectation]]，无需这些 indicators 独立。

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
