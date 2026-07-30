---
aliases: [Graph Tree, Tree, Tree in Graph Theory, Forest Graph, Leaf Vertex, 图论中的树, 森林, 叶子顶点]
tags: [concept, discrete-mathematics, graph-theory]
---
# Graph Tree

有限无向图是树，若它连通且无环。对含 $n$ 个顶点的有限简单图，以下条件等价：

- 连通且无环；
- 任意两顶点间恰有一条简单路径；
- 连通且有 $n-1$ 条边；
- 无环且有 $n-1$ 条边。

当 $n\ge2$ 时，树至少有两个叶子。

forest 是不要求连通的无环图；每个 connected component 都是一棵树。leaf 是 degree 为 $1$ 的顶点（单顶点树是否把唯一顶点称作 leaf 取决于约定）。

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
