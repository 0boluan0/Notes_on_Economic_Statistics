---
aliases: [Fermat's Little Theorem, FLT, 费马小定理]
tags: [concept, discrete-mathematics, number-theory]
type: concept
---
# Fermat's Little Theorem

若 $p$ 是素数且 $p\nmid a$，则

$$
a^{p-1}\equiv1\pmod p.
$$

等价地，对任意整数 $a$ 都有 $a^p\equiv a\pmod p$。第一种形式不能漏掉 $p\nmid a$；反过来，某个数满足该同余也不足以单独证明模数是素数。

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
