---
aliases: [Asymptotic Notation, Big O, Big Omega, Big Theta, 渐近记号]
tags: [concept, discrete-mathematics, algorithms]
---
# Asymptotic Notation

对最终非负函数 $f,g$：

- $f=O(g)$：存在 $c>0,n_0$，使 $n\ge n_0$ 时 $f(n)\le cg(n)$；
- $f=\Omega(g)$：存在 $c>0,n_0$，使 $f(n)\ge cg(n)$；
- $f=\Theta(g)$：同时为 $O(g)$ 与 $\Omega(g)$。

这些是函数集合关系，不是普通等式。有限前缀和常数因子不影响渐近类，但振荡函数可能彼此不可比较。

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
