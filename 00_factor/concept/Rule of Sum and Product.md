---
aliases: [Rule of Sum and Product, Sum and Product Rules, 加法与乘法计数法则]
tags: [concept, discrete-mathematics, counting]
---
# Rule of Sum and Product

若对象集合被拆成互不相交的 $A_1,\ldots,A_k$，则
$$
\left|\bigcup_i A_i\right|=\sum_i|A_i|.
$$
若一个对象由依次选择 $x_1\in A_1,\ldots,x_k\in A_k$ 唯一确定，且每步选择数不受先前具体选择影响，则总数为 $\prod_i|A_i|$。

加法要求分支互斥；乘法要求每个最终对象恰对应一条选择路径。

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
