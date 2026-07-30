---
aliases: [Cardinality, Set Cardinality, 基数, 集合基数]
tags: [concept, discrete-mathematics, set-theory]
type: concept
---
# Cardinality

集合 $A$ 的基数 $|A|$ 表示其元素数量。有限集合可直接计数；一般地，若存在双射 $A\leftrightarrow B$，就定义 $A$ 与 $B$ 等势，即 $|A|=|B|$。

对无限集合，“真子集”仍可能与原集合等势，例如 $n\mapsto2n$ 给出 $\mathbb N\leftrightarrow2\mathbb N$。因此无限基数比较必须使用 injection、surjection 或 bijection，而不能只凭包含关系判断。

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
