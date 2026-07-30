---
aliases: [Divisibility, Divides Relation, 整除, 整除关系]
tags: [concept, discrete-mathematics, number-theory]
type: concept
---
# Divisibility

对整数 $a,b$，若存在整数 $k$ 使 $b=ak$，就称 $a$ 整除 $b$，记作 $a\mid b$。否则记作 $a\nmid b$。

整除具有传递性；若 $a\mid b$ 且 $a\mid c$，则 $a\mid (xb+yc)$ 对任意整数 $x,y$ 成立。$0\mid b$ 当且仅当 $b=0$，而每个整数都整除 $0$。

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
