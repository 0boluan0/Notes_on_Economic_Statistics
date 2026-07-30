---
aliases: [Euler Totient Theorem, Euler's Totient Theorem, Euler's Theorem, 欧拉定理]
tags: [concept, discrete-mathematics, number-theory]
---
# Euler Totient Theorem

若 $\gcd(a,n)=1$，则
$$
a^{\varphi(n)}\equiv1\pmod n.
$$
当 $n=p$ 为素数时得到 Fermat 小定理 $a^{p-1}\equiv1\pmod p$（前提 $p\nmid a$）。

定理的前提不能省略；例如 $a=2,n=4$ 时 $2^{\varphi(4)}=4\not\equiv1\pmod4$。

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
