---
aliases: [Graph Matching, Matching in Graph Theory, 图匹配, 匹配]
tags: [concept, discrete-mathematics, graph-theory]
type: concept
---
# Graph Matching

图中的 matching 是一组两两不共享端点的边。若 matching 覆盖每个顶点，就称 perfect matching；若只要求边数达到最大，称 maximum matching；若按边权优化，则是 maximum/minimum-weight matching。

maximum 表示全局边数最多，maximal 只表示当前 matching 不能再加入一条边；极大 matching 不一定最大。[[Stable Matching|稳定匹配]]还带双方偏好与 blocking-pair 条件，是不同的优化标准。

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
