---
aliases: [Walk Path and Cycle, Walk, Path, Cycle, 游走路径与环]
tags: [concept, discrete-mathematics, graph-theory]
---
# Walk Path and Cycle

walk 是相邻顶点序列，允许重复顶点和边；path 通常指不重复顶点的 walk；cycle 是首尾相同、内部顶点不重复的闭合路径。

可达性只要求存在某条 walk；任何连接两个顶点的 walk 都可删去重复环得到 path。定义必须同时注明图是有向还是无向。

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
