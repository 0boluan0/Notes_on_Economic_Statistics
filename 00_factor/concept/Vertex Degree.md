---
aliases: [Vertex Degree, Graph Degree, Handshake Lemma, Handshaking Lemma, 顶点度数, 握手定理]
tags: [concept, discrete-mathematics, graph-theory]
---
# Vertex Degree

无向图中顶点 $v$ 的度数 $\deg(v)$ 是与 $v$ 相接的边数，self-loop 通常贡献 $2$。有限无向图满足握手定理
$$
\sum_{v\in V}\deg(v)=2|E|,
$$
所以奇度顶点个数必为偶数。

有向图需区分入度与出度，二者在全图上的总和都等于边数。

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
