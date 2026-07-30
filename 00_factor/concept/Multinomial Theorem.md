---
aliases: [Multinomial Theorem, Multinomial Coefficient, 多项式定理, 多项式系数]
tags: [concept, discrete-mathematics, counting]
type: concept
---
# Multinomial Theorem

对非负整数 $n$，

$$
(x_1+\cdots+x_m)^n
=\sum_{k_1+\cdots+k_m=n}
\frac{n!}{k_1!\cdots k_m!}x_1^{k_1}\cdots x_m^{k_m}.
$$

系数 $\binom{n}{k_1,\ldots,k_m}=n!/(k_1!\cdots k_m!)$ 计数把 $n$ 个有标号位置分进 $m$ 个有标号组、各组大小为 $k_i$ 的方式。

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
