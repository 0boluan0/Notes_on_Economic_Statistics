---
aliases: [Graph Coloring, Vertex Coloring, Chromatic Number, 图着色, 色数]
tags: [concept, discrete-mathematics, graph-theory]
---
# Graph Coloring

proper vertex coloring 给每个顶点赋颜色，使每条边的两个端点颜色不同。图的 chromatic number $\chi(G)$ 是所需最少颜色数。

二分图当且仅当没有奇环，也当且仅当可用两色 proper coloring。贪心着色给出上界，但通常不保证最少颜色。

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
