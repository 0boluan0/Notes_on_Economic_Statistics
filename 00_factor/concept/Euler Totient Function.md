---
aliases: [Euler Totient Function, Euler Phi Function, 欧拉函数]
tags: [concept, discrete-mathematics, number-theory]
---
# Euler Totient Function

对正整数 $n$，$\varphi(n)$ 是 $1,2,\ldots,n$ 中与 $n$ 互素的整数个数。若
$$
n=\prod_i p_i^{\alpha_i},
$$
则
$$
\varphi(n)=n\prod_i\left(1-\frac1{p_i}\right).
$$

特别地，素数 $p$ 满足 $\varphi(p)=p-1$；不同素数 $p,q$ 满足 $\varphi(pq)=(p-1)(q-1)$。

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
