---
aliases: [Chinese Remainder Theorem, CRT, 中国剩余定理, 孙子定理]
tags: [concept, discrete-mathematics, number-theory]
type: concept
---
# Chinese Remainder Theorem

若整数模数 $n_1,\ldots,n_k\ge2$ 两两互素，则对任意余数 $a_1,\ldots,a_k$，同余方程组

$$
x\equiv a_i\pmod{n_i}\qquad(i=1,\ldots,k)
$$

存在解，且解在模 $N=\prod_i n_i$ 意义下唯一。若模数不两两互素，则需额外检查余数在各最大公因数模下是否相容，不能直接套用该版本。

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
