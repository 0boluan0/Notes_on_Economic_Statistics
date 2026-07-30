---
aliases: [Topological Sort, Topological Ordering, 拓扑排序]
tags: [procedure, discrete-mathematics, graph-theory]
type: procedure
---
# Topological Sort

1. 计算每个顶点入度，把所有入度为零的顶点加入候选集合。
2. 取出一个候选顶点并输出。
3. 删除它的所有出边；新变为入度零的顶点加入候选集合。
4. 重复直到候选集合为空。
5. 若输出了全部顶点，所得序列是拓扑顺序；否则剩余子图含有向环。

不同候选选择可能给出不同但都合法的顺序。

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
