---
aliases: [Proof by Contradiction, 反证法]
tags: [procedure, discrete-mathematics]
type: procedure
---
# Proof by Contradiction

1. 明确目标命题 $P$ 及其精确否定 $\neg P$。
2. 暂时假设 $\neg P$。
3. 仅从该假设、定义与已知事实推导出 $R\land\neg R$、违反定义或违反已知定理的结论。
4. 因 $\neg P$ 不可能，得到 $P$。

反证法的关键不是写出“矛盾”二字，而是指出哪两个陈述不能同时为真。

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
