---
aliases: [Graph Adjacency Matrix, Adjacency Matrix, 图的邻接矩阵, 邻接矩阵]
tags: [concept, discrete-mathematics, graph-theory]
type: concept
---
# Graph Adjacency Matrix

给顶点编号 $v_1,\ldots,v_n$。simple graph 或 simple directed graph 的邻接矩阵满足：有边 $v_i\to v_j$ 时 $A_{ij}=1$，否则为 $0$；若允许平行边，则相应 multigraph 邻接矩阵可用 $A_{ij}$ 记录边数。无向图的 $A$ 对称。

对允许重复顶点的 directed walks，$(A^k)_{ij}$ 等于从 $v_i$ 到 $v_j$ 的长度 $k$ walk 数。矩阵依赖顶点编号；同时置换行和列不会改变图的同构类型。

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
