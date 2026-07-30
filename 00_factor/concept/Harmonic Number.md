---
aliases: [Harmonic Number, Harmonic Numbers, 调和数]
tags: [concept, discrete-mathematics, series]
type: concept
---
# Harmonic Number

第 $n$ 个 harmonic number 定义为

$$
H_n=\sum_{k=1}^{n}\frac1k.
$$

它满足 $\ln(n+1)\le H_n\le1+\ln n$，因此 $H_n=\Theta(\ln n)$ 且趋于无穷。更精细地，$H_n=\ln n+\gamma+o(1)$，其中 $\gamma$ 是 Euler–Mascheroni constant。

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
