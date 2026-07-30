---
aliases: [Graph Isomorphism, 图同构]
tags: [concept, discrete-mathematics, graph-theory]
---
# Graph Isomorphism

图 $G,H$ 同构，若存在顶点双射 $f:V(G)\to V(H)$，使邻接关系被保持：
$$
\{u,v\}\in E(G)\Longleftrightarrow\{f(u),f(v)\}\in E(H).
$$
同构图只是顶点改名，具有相同的度数多重集、连通分量、环结构和着色数。

这些不变量不同可证明不同构；相同通常不足以证明同构，仍需给出明确映射。

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
