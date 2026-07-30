---
aliases: [Minimum Spanning Tree, MST, 最小生成树]
tags: [concept, discrete-mathematics, graph-theory, algorithms]
type: concept
---
# Minimum Spanning Tree

在边带实数权重的有限连通无向图中，minimum spanning tree 是总边权最小的 [[Spanning Tree|生成树]]。最小生成树一定存在，但当边权有并列时不一定唯一。

“minimum”优化的是整棵树的总权重，不表示每一步都取当前图中绝对最轻的边；Kruskal 与 Prim 的贪心选择依赖 cut/cycle 性质保证全局正确性。

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
