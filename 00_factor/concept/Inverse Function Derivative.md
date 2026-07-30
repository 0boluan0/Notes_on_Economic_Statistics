---
aliases: [Inverse Function Derivative, 反函数导数, 反函数求导定理]
tags: [concept, calculus]
---
# Inverse Function Derivative

若 $f$ 在 $a$ 附近可逆、$f'(a)\ne0$，令 $b=f(a)$，则

$$
(f^{-1})'(b)=\frac{1}{f'(a)}=\frac{1}{f'(f^{-1}(b))}.
$$

原函数与反函数图像关于 $y=x$ 对称，对应切线斜率互为倒数。

## 最小例子

由 $f(x)=e^x$ 得 $(\ln x)'=1/x$，其中 $x>0$。

## 关联卡片

- [[Derivative]]
- [[Natural Logarithm]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
