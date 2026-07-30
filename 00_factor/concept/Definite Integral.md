---
aliases: [Definite Integral, 定积分, 有向面积]
tags: [concept, calculus]
---
# Definite Integral

定积分是 Riemann 和在分割加细时的极限：

$$
\int_a^b f(x)\,dx
=\lim_{\|P\|\to0}\sum_i f(x_i^*)\Delta x_i.
$$

它表示带符号的累计量；积分值是数，不是一个函数族。

## 最小例子

$\int_0^1x\,dx=1/2$。

## 关联卡片

- [[Riemann Sum]]
- [[Integral]]
- [[Fundamental Theorem of Calculus]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
