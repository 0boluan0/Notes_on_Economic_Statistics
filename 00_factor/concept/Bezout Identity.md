---
aliases: [Bezout Identity, Bézout's Identity, 贝祖等式]
tags: [concept, discrete-mathematics, number-theory]
---
# Bezout Identity

对整数 $a,b$（不全为零），存在整数 $x,y$ 使
$$
ax+by=\gcd(a,b).
$$
全部线性组合 $ax+by$ 恰是 $\gcd(a,b)$ 的整数倍。

因此同余方程 $ax\equiv1\pmod m$ 有解，当且仅当 $\gcd(a,m)=1$；此时 Bézout 系数给出 [[Modular Inverse]]。

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
