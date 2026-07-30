---
aliases: [Greatest Common Divisor, GCD, 最大公约数]
tags: [concept, discrete-mathematics, number-theory]
---
# Greatest Common Divisor

对不全为零的整数 $a,b$，$\gcd(a,b)$ 是同时整除 $a,b$ 的最大正整数。它也等于全部整数线性组合 $ax+by$ 中最小的正值，并可由 [[Euclidean Algorithm]] 计算。

特别地，$\gcd(a,b)=1$ 称 $a,b$ 互素；这等价于存在整数 $x,y$ 使 $ax+by=1$，见 [[Bezout Identity]]。

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
