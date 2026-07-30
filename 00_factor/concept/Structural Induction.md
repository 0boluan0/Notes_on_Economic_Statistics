---
aliases: [Structural Induction, 结构归纳法]
tags: [concept, discrete-mathematics]
---
# Structural Induction

结构归纳用于由 [[Recursive Definition]] 生成的对象。若性质 $P$ 对每个基础对象成立，并且每个构造规则在输入对象满足 $P$ 时保证新对象也满足 $P$，则所有有限生成的对象都满足 $P$。

它与对“构造树高度”做数学归纳等价。证明必须逐一覆盖递归定义中的全部 constructor。

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
