---
aliases: [Euclidean Algorithm, Euclid's Algorithm, 欧几里得算法, 辗转相除法]
tags: [procedure, discrete-mathematics, number-theory]
type: procedure
---
# Euclidean Algorithm

## 输入与输出

输入整数 $a,b$（不全为零），输出 $\gcd(a,b)$；反向代入还可输出 Bézout 系数。

1. 取绝对值并令 $a\ge b>0$。
2. 重复写 $a=qb+r$，其中 $0\le r<b$。
3. 用 $(b,r)$ 替换 $(a,b)$，直到余数为 $0$。
4. 最后一个非零余数是 gcd。
5. 若需要系数，从最后一行开始反向代入。

依据是 $\gcd(a,b)=\gcd(b,a-qb)$；余数严格下降保证终止。

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
