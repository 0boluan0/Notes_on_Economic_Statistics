---
aliases: [Discrete Probability Space, 离散概率空间]
tags: [concept, discrete-mathematics, probability]
---
# Discrete Probability Space

离散概率空间由可数样本空间 $\Omega$ 和质量函数 $p:\Omega\to[0,1]$ 构成，满足
$$
\sum_{\omega\in\Omega}p(\omega)=1.
$$
事件 $A\subseteq\Omega$ 的概率为 $\Pr(A)=\sum_{\omega\in A}p(\omega)$。

只有当 outcomes 等可能时，才能使用 $\Pr(A)=|A|/|\Omega|$。

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
