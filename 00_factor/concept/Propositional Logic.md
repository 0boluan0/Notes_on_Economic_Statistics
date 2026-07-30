---
aliases: [Propositional Logic, 命题逻辑]
tags: [concept, discrete-mathematics]
---
# Propositional Logic

命题逻辑研究具有确定真值的命题，以及由 $\neg,\land,\lor,\Rightarrow,\Leftrightarrow$ 组成的公式。公式的真值只由原子命题的真值和连接词语义决定。

## 最小例子

$P\Rightarrow Q$ 只在 $P$ 真而 $Q$ 假时为假；因此它等价于 $\neg P\lor Q$。

## 边界

命题逻辑不描述对象内部结构；含变量和量词的断言属于 [[Predicate Logic and Quantifiers]]。

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
