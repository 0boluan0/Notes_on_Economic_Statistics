---
aliases: [Cantor-Schroeder-Bernstein Theorem, Schroeder-Bernstein Theorem, Cantor–Schröder–Bernstein Theorem, 康托尔-施罗德-伯恩斯坦定理]
tags: [concept, discrete-mathematics, set-theory]
type: concept
---
# Cantor-Schroeder-Bernstein Theorem

若存在 injections $f:A\to B$ 与 $g:B\to A$，则存在 bijection $h:A\to B$。因此证明两个集合等势时，可以分别构造两个方向的单射，而不必直接写出双射。

定理只保证某个双射存在，不能把原来的两个单射简单拼接为双射。把“两个方向都有 surjection”改写成同类结论时还需要从各原像集合选择代表；在不预设选择公理的语境中，不能把它无条件称作等价版本。

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
