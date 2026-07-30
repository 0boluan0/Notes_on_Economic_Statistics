---
aliases: [Mathematical Induction, Strong Induction, 数学归纳法, 普通归纳法, 强归纳法]
tags: [concept, discrete-mathematics]
---
# Mathematical Induction

设 $P(n)$ 是对 $n\ge n_0$ 的命题。若 $P(n_0)$ 成立，且对每个 $n\ge n_0$ 都有 $P(n)\Rightarrow P(n+1)$，则 $P(n)$ 对全部 $n\ge n_0$ 成立。

强归纳把归纳假设改为 $P(n_0),\ldots,P(n)$ 全部成立，再推出 $P(n+1)$；它与普通归纳逻辑等价，但适合对象依赖多个更小规模的情形。

执行模板见 [[Induction Proof Procedure]]。

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
