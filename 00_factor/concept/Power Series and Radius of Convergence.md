---
aliases: [Power Series and Radius of Convergence, Power Series, Radius of Convergence, 幂级数, 收敛半径]
tags: [concept, calculus]
---
# Power Series and Radius of Convergence

以 $a$ 为中心的幂级数为

$$
\sum_{n=0}^{\infty}c_n(x-a)^n.
$$

存在收敛半径 $R\in[0,\infty]$：当 $|x-a|<R$ 时绝对收敛，当 $|x-a|>R$ 时发散，端点需另查。

## 最小例子

几何级数 $\sum_{n=0}^{\infty}x^n$ 在 $|x|<1$ 收敛，因此 $R=1$。

## 关联卡片

- [[Series]]
- [[Taylor Expansion]]
- [[Series Convergence Tests]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
