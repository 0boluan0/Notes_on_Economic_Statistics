---
aliases: [Spanning Tree, 生成树]
tags: [concept, discrete-mathematics, graph-theory]
---
# Spanning Tree

连通图 $G$ 的 spanning tree 是包含 $G$ 全部顶点且边集为 $E(G)$ 子集的一棵树。任何有限连通图都可通过不断删除环上的边得到生成树。

生成树保留连通性并使用最少边数 $|V|-1$；它不一定唯一。若图不连通，对每个分量取树得到 spanning forest。

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
