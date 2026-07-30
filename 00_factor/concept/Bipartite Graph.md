---
aliases: [Bipartite Graph, Bipartition, 二分图, 二部图]
tags: [concept, discrete-mathematics, graph-theory]
type: concept
---
# Bipartite Graph

图 $G=(V,E)$ 是 bipartite，若顶点可分成互不相交的两侧 $L,R$，且每条边都连接一侧的顶点与另一侧的顶点。等价地，图可 proper 2-color；对有限无向图，也等价于没有 odd cycle。

bipartition 可能不唯一；若图连通且至少含一条边，则交换两侧以外是唯一的。matching 与 stable matching 通常都在 bipartite graph 上表述，但两者的目标条件不同。

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
