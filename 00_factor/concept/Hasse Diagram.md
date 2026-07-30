---
aliases: [Hasse Diagram, Hasse 图]
tags: [concept, discrete-mathematics, order-theory]
type: concept
---
# Hasse Diagram

有限 poset 的 Hasse 图只画 cover relations：当 $x\prec y$ 且不存在 $z$ 满足 $x\prec z\prec y$ 时连接 $x,y$，并把较大的元素画在上方。图中省略自环、传递边与向上箭头。

从 $x$ 沿边持续向上能到达 $y$，当且仅当 $x\preceq y$。若画图没有固定“向上”约定，就必须显式标方向。

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
