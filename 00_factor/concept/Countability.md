---
aliases: [Countability, Countable Set, 可数性, 可数集合]
tags: [concept, discrete-mathematics]
---
# Countability

集合 $A$ 是可数的，若它是有限集，或存在从 $\mathbb N$ 到 $A$ 的双射。可数无限意味着元素可以排成不遗漏、无重复的序列 $a_0,a_1,\ldots$。

$\mathbb Z$ 与 $\mathbb Q$ 可数；$\mathcal P(\mathbb N)$ 和 $\mathbb R$ 不可数。不可数性的典型工具是 [[Cantor Diagonal Argument]]。

## 易错点

“稠密”不等于“不可数”：$\mathbb Q$ 在 $\mathbb R$ 中稠密但仍可数。

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
