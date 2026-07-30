---
aliases: [Chain and Antichain, Chain, Antichain, 链与反链, 链, 反链]
tags: [concept, discrete-mathematics, order-theory]
type: concept
---
# Chain and Antichain

在 poset $(P,\preceq)$ 中，**chain** 是任意两个元素都可比较的子集；**antichain** 是任意两个不同元素都不可比较的子集。chain 描述必须串行的依赖，antichain 描述可并行的一组对象。

“极大（maximal）chain”指按包含关系不能再扩张；“最长（maximum-cardinality）chain”指元素数达到全局最大。两者不能混用。

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
