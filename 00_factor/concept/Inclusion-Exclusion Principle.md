---
aliases: [Inclusion-Exclusion Principle, PIE, 容斥原理]
tags: [concept, discrete-mathematics, counting]
---
# Inclusion-Exclusion Principle

有限集合 $A_1,\ldots,A_n$ 的并集大小为
$$
\left|\bigcup_iA_i\right|
=\sum_i|A_i|-\sum_{i<j}|A_i\cap A_j|+\sum_{i<j<k}|A_i\cap A_j\cap A_k|-\cdots.
$$

一个属于恰好 $r$ 个集合的元素在右侧总系数为
$$
\binom r1-\binom r2+\cdots+(-1)^{r+1}\binom rr=1,
$$
所以最终恰被计一次。

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
