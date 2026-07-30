---
aliases: [Markov Inequality, Markov Bound, 马尔可夫不等式]
tags: [concept, discrete-mathematics, probability]
---
# Markov Inequality

若 $X\ge0$ 且 $a>0$，则
$$
\Pr(X\ge a)\le\frac{\mathbb E[X]}{a}.
$$

它只用非负性与均值，适用广但可能很松。若变量可取负值，不能直接套用；可对非负函数如 $(X-\mu)^2$ 应用，从而得到 [[Chebyshev Inequality]]。

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
