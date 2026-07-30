---
aliases: [Hyperbolic Functions, Hyperbolic Trigonometric Functions, 双曲函数, 双曲三角函数]
tags: [concept, calculus]
---
# Hyperbolic Functions

双曲正弦与双曲余弦定义为

$$
\sinh x=\frac{e^x-e^{-x}}{2},\qquad \cosh x=\frac{e^x+e^{-x}}{2}.
$$

它们满足 $\cosh^2x-\sinh^2x=1$，并且 $(\sinh x)'=\cosh x$、$(\cosh x)'=\sinh x$。

## 最小例子

$x=\cosh t$、$y=\sinh t$ 参数化双曲线 $x^2-y^2=1$ 的右支。

## 关联卡片

- [[Exponential Function]]
- [[Parametric Curve]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
