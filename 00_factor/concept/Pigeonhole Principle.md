---
aliases: [Pigeonhole Principle, Dirichlet Box Principle, 鸽巢原理]
tags: [concept, discrete-mathematics, counting]
---
# Pigeonhole Principle

把 $n\ge0$ 个对象放进 $k\ge1$ 个盒子，至少有一个盒子含有
$$
\left\lceil\frac nk\right\rceil
$$
个对象。特别地，若 $n>k$，至少两个对象进入同一盒。

应用的难点是选择对象、盒子和映射；原理只保证碰撞存在，不指出发生在哪个盒子。

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
