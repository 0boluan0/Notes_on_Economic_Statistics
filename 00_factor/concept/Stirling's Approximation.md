---
aliases: [Stirling's Approximation, Stirling's Formula, Stirling Formula, 斯特林公式, 斯特林近似]
tags: [concept, discrete-mathematics, asymptotics]
type: concept
---
# Stirling's Approximation

当 $n\to\infty$ 时，

$$
n!\sim\sqrt{2\pi n}\left(\frac ne\right)^n.
$$

它给出 factorial 的相对误差趋零近似；只写 $(n/e)^n$ 会漏掉不可忽略的 $\sqrt{n}$ 因子。常用对数形式是

$$
\ln(n!)=n\ln n-n+\tfrac12\ln(2\pi n)+o(1).
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
