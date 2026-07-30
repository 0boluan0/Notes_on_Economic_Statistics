---
aliases: [Predicate Logic and Quantifiers, Predicate Logic, Predicate, 谓词逻辑, 谓词, 量词]
tags: [concept, discrete-mathematics]
---
# Predicate Logic and Quantifiers

谓词 $P(x)$ 在指定论域中接收对象并产生真值。全称量词 $\forall x$ 表示论域中每个对象都满足，存在量词 $\exists x$ 表示至少有一个对象满足。

量词否定遵循
$$
\neg\forall x\,P(x)\equiv\exists x\,\neg P(x),\qquad
\neg\exists x\,P(x)\equiv\forall x\,\neg P(x).
$$

量词次序通常不可交换；$\forall x\exists y,R(x,y)$ 允许 $y$ 依赖 $x$，而 $\exists y\forall x,R(x,y)$ 要求同一个 $y$ 对所有 $x$ 有效。

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
