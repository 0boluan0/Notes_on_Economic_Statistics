---
aliases: [Proof by Cases, 分类证明, 分情况证明]
tags: [procedure, discrete-mathematics]
type: procedure
---
# Proof by Cases

1. 选取覆盖全部允许输入的情形 $C_1,\ldots,C_k$。
2. 说明 $C_1\lor\cdots\lor C_k$ 恒成立。
3. 在每个 $C_i$ 下独立推出同一目标 $Q$。
4. 汇总得到 $Q$ 对全部输入成立。

情形可以重叠，但若用于计数，重叠会造成重复计数，必须改为互斥划分或使用 [[Inclusion-Exclusion Principle]]。

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
