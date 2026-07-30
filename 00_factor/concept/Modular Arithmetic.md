---
aliases: [Modular Arithmetic, Congruence, 模运算, 同余]
tags: [concept, discrete-mathematics, number-theory]
---
# Modular Arithmetic

对 $m>0$，
$$
a\equiv b\pmod m\quad\Longleftrightarrow\quad m\mid(a-b).
$$
同余关系把整数按相同余数分成等价类。加法、减法、乘法和非负整数次幂保持同余；除法只有在被约去因子对模数可逆时才合法。

## 易错点

由 $ac\equiv bc\pmod m$ 不能无条件推出 $a\equiv b\pmod m$；需要 $\gcd(c,m)=1$，或把模数相应缩小。

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
