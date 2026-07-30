---
aliases: [Graph Connectivity, Connected Graph, Connected Component, 图连通性, 连通分量]
tags: [concept, discrete-mathematics, graph-theory]
---
# Graph Connectivity

无向图是连通的，若任意两顶点之间都有 path。极大连通子图称 connected component。

删除后增加连通分量数的边称 bridge；删除后增加连通分量数的顶点称 articulation vertex。连通图总含 [[Spanning Tree]]。

有向图需区分强连通与忽略方向后的弱连通。

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
