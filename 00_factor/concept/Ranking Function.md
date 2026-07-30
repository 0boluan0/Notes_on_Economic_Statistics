---
aliases: [Ranking Function, Derived Variable for Termination, 排名函数, 终止度量]
tags: [concept, discrete-mathematics, termination]
type: concept
---
# Ranking Function

排名函数把每个尚未终止的状态映到一个良基序集合，且每次有效转移都使该值严格下降。因为良基序不存在无限严格下降链，执行不可能永远继续，因此必须终止。

最常用的值域是 $\mathbb N$。只证明“值不增加”不足以推出终止；还必须严格下降，并说明值始终留在良基集合中。

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
