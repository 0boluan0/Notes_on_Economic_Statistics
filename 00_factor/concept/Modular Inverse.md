---
aliases: [Modular Inverse, 模逆元, 乘法逆元]
tags: [concept, discrete-mathematics, number-theory]
---
# Modular Inverse

对整数 $m\ge2$，若存在 $x$ 使 $ax\equiv1\pmod m$，则 $x$ 是 $a$ 模 $m$ 的乘法逆元，记作 $a^{-1}\pmod m$。

逆元存在当且仅当 $\gcd(a,m)=1$；存在时在模 $m$ 意义下唯一。可用 [[Euclidean Algorithm]] 的反向代入求得。

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
