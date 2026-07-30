---
aliases: [Random Walk on a Graph, Graph Random Walk, 图上随机游走]
tags: [concept, discrete-mathematics, probability, graph-theory]
---
# Random Walk on a Graph

图上随机游走以顶点为状态，每一步按当前顶点的邻边概率选择下一个顶点。简单无向图上的 simple random walk 从 $v$ 到每个邻点的概率为 $1/\deg(v)$。

它形成 [[Markov Chain]]。有限连通非二分无向图上，分布收敛到
$$
\pi(v)=\frac{\deg(v)}{2|E|}.
$$
连通性关系到不可约性，二分性关系到周期性；二者职责不能混淆。

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
