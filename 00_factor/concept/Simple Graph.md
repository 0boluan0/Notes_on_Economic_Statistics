---
aliases: [Simple Graph, 简单图]
tags: [concept, discrete-mathematics, graph-theory]
type: concept
---
# Simple Graph

simple graph $G=(V,E)$ 的每条边是两个不同顶点组成的无序集合，因此没有 self-loop，也没有 parallel edges。若边有方向，应改用 [[Directed Graph|directed graph]]；若允许重复边，则是 multigraph。

对含 $n$ 个顶点的 simple graph，每个顶点度数至多 $n-1$，边数至多 $\binom n2$。

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
