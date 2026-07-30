---
aliases: [Markov Inequality Proof, 马尔可夫不等式证明]
tags: [proof, discrete-mathematics, probability]
type: proof
---
# Markov Inequality Proof

若 $X\ge0$ 且 $a>0$，逐点有
$$
X\ge aI_{\{X\ge a\}}.
$$
对两边取期望并使用单调性：
$$
\mathbb E[X]\ge a\mathbb E[I_{\{X\ge a\}}]
=a\Pr(X\ge a).
$$
除以正数 $a$ 得
$$
\Pr(X\ge a)\le\frac{\mathbb E[X]}a.
$$

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
