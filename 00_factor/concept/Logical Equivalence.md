---
aliases: [Logical Equivalence, 逻辑等价]
tags: [concept, discrete-mathematics, logic]
type: concept
---
# Logical Equivalence

命题公式 $P$ 与 $Q$ 逻辑等价，记作 $P\equiv Q$，若它们在每一种变量赋值下真值都相同；等价地，$P\Leftrightarrow Q$ 是重言式。

例如 $P\Rightarrow Q\equiv\neg P\lor Q$，以及 $\neg(P\land Q)\equiv\neg P\lor\neg Q$。逻辑等价允许在证明中双向替换；单向蕴含不允许这样做。

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
