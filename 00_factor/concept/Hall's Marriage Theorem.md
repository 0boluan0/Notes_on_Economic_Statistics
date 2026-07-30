---
aliases: [Hall's Marriage Theorem, Hall's Theorem, Hall Condition, 霍尔婚配定理, Hall 条件]
tags: [concept, discrete-mathematics, graph-theory, matching]
type: concept
---
# Hall's Marriage Theorem

对有限 bipartite graph $G=(L\cup R,E)$，存在一个 matching 覆盖 $L$ 的每个顶点，当且仅当对所有 $S\subseteq L$ 都有

$$
|N(S)|\ge |S|,
$$

其中 $N(S)$ 是 $S$ 在 $R$ 一侧的邻居集合。若 $|L|=|R|$，这样的 matching 就是 perfect matching。

Hall 条件必须检查任意子集而不只是单个顶点；失败的 $S$ 是一个 bottleneck certificate，说明 $|S|$ 个左侧对象只能争夺更少的右侧对象。

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
