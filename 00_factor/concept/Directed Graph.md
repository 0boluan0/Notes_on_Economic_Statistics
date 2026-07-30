---
aliases: [Directed Graph, Digraph, 有向图]
tags: [concept, discrete-mathematics, graph-theory]
---
# Directed Graph

有向图 $G=(V,E)$ 由顶点集合 $V$ 和有序对边集 $E\subseteq V\times V$ 构成。边 $(u,v)$ 从 $u$ 指向 $v$，一般不蕴含反向边 $(v,u)$。

有向图适合描述先决条件、状态转移、网页链接和程序控制流。序列化结构见 [[Walk Path and Cycle]]；无环情形见 [[Directed Acyclic Graph]]。

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
