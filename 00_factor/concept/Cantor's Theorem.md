---
aliases: [Cantor's Theorem, Cantor Theorem, 康托尔定理]
tags: [concept, discrete-mathematics, set-theory]
type: concept
---
# Cantor's Theorem

对任意集合 $A$，其幂集 $\mathcal P(A)$ 的基数严格大于 $A$：存在 injection $A\to\mathcal P(A)$，但不存在 surjection $A\to\mathcal P(A)$。

严格小于同时包含两层信息：singleton map 给出 $A\to\mathcal P(A)$ 的 injection，而 [[Cantor Diagonal Argument|对角论证]]排除反方向所需的 surjection。

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
