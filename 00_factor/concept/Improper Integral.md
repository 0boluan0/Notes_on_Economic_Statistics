---
aliases: [Improper Integral, 反常积分, 广义积分]
tags: [concept, calculus]
---
# Improper Integral

积分区间无界或被积函数在区间内无界时，积分通过极限定义。例如

$$
\int_a^\infty f(x)\,dx=\lim_{R\to\infty}\int_a^R f(x)\,dx.
$$

只有相应极限存在且有限时才称积分收敛；内部奇点必须拆区间分别检查。

## 最小例子

$\int_1^\infty x^{-p}\,dx$ 当且仅当 $p>1$ 收敛。

## 关联卡片

- [[Definite Integral]]
- [[Limit]]
- [[Series Convergence Tests]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
