---
aliases: [Equivalence Relation, 等价关系]
tags: [concept, discrete-mathematics]
---
# Equivalence Relation

集合上的等价关系满足自反、对称和传递性。每个元素 $a$ 的等价类为
$$
[a]=\{x:x\sim a\}.
$$
不同等价类要么相同、要么不交，并共同构成原集合的一个 partition；反过来，每个 partition 都定义一个等价关系。

整数模 $m$ 同余是基本例子。

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
