---
aliases: [Variance, 方差]
tags: [concept, discrete-mathematics, probability]
---
# Variance

若 $\mu=\mathbb E[X]$，方差定义为
$$
\operatorname{Var}(X)=\mathbb E[(X-\mu)^2]
=\mathbb E[X^2]-\mu^2.
$$
标准差为 $\sqrt{\operatorname{Var}(X)}$，与 $X$ 单位相同。

$\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X)$。只有在独立或协方差项为零时，和的方差才直接相加。

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
