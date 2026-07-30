---
aliases: [Chain Rule, 链式法则, 复合函数求导法则]
tags: [concept, calculus]
---
# Chain Rule

若 $y=f(u)$、$u=g(x)$，且相应导数存在，则复合函数满足

$$
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx},\qquad (f\circ g)'(x)=f'(g(x))g'(x).
$$

它表达“外层对中间变量的变化率”与“中间变量对输入的变化率”相乘。

## 最小例子

$(\sin x^2)'=\cos(x^2)\cdot2x$。

## 关联卡片

- [[Derivative]]
- [[Implicit Differentiation]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
