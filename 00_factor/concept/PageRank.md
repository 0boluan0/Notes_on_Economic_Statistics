---
aliases: [PageRank, PageRank Algorithm, 网页排名算法]
tags: [concept, discrete-mathematics, probability, graph-theory]
---
# PageRank

PageRank 把网页看作有向图上的随机游走，并加入 teleportation。采用行概率向量约定时，先把 dangling pages 修补为合法转移，使 $P$ 为行随机矩阵；若 $0<\alpha<1$、$v$ 是 teleportation 概率向量，则 PageRank $\pi$ 满足
$$
\pi=\alpha\pi P+(1-\alpha)v.
$$

若使用列概率向量，等价公式是 $\pi=\alpha P\pi+(1-\alpha)v$，此时 $P$ 应为列随机矩阵。若 $v$ 的各分量均为正，teleportation 会处理 spider traps 与不可约/周期问题，使有限 Markov chain 拥有唯一且可由迭代得到的平稳分布。

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
