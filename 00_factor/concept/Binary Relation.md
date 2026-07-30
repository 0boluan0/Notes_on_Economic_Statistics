---
aliases: [Binary Relation, 二元关系]
tags: [concept, discrete-mathematics]
---
# Binary Relation

从 $A$ 到 $B$ 的二元关系是 $A\times B$ 的任意子集 $R$；写作 $aRb$ 表示 $(a,b)\in R$。函数是满足每个输入恰好关联一个输出的特殊关系。

定义在同一集合上的关系常考察自反、对称、反对称和传递性；不同性质组合产生 [[Partial Order]] 与 [[Equivalence Relation]]。

## 最小例子

整数上的“整除”关系是自反、反对称、传递的；“同余模 $m$”是自反、对称、传递的。

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
