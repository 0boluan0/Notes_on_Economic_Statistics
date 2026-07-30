---
aliases: [Handshake Lemma Proof, 握手定理证明]
tags: [proof, discrete-mathematics, graph-theory]
type: proof
---
# Handshake Lemma Proof

对有限无向图的“顶点—边端点出现”作双重计数。令
$$
I=\{(v,e,s):v\text{ 是边 }e\text{ 的第 }s\text{ 个端点}\}.
$$
按顶点计数得到 $|I|=\sum_v\deg(v)$；按边计数，每条边贡献两个端点出现，因此 $|I|=2|E|$。self-loop 的两个端点虽落在同一顶点，仍是两次出现。两式相等即得握手定理。

因为度数和为偶数，奇数个奇数相加才为奇数，所以奇度顶点的数量必须为偶数。

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
