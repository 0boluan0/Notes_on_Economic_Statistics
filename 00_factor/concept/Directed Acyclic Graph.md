---
aliases: [Directed Acyclic Graph, DAG, 有向无环图]
tags: [concept, discrete-mathematics, graph-theory]
---
# Directed Acyclic Graph

DAG 是没有有向环的 [[Directed Graph]]。有限有向图是 DAG，当且仅当它存在拓扑顺序；也等价于每个非空诱导子图都有入度为零的顶点。

DAG 用于依赖、调度和构建系统。最长有向路径给出无限处理器下仍无法突破的关键路径时间。

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
