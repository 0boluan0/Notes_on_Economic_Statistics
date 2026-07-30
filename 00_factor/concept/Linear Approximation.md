---
aliases: [Linear Approximation, Linearization, 线性近似, 线性化]
tags: [concept, calculus]
---
# Linear Approximation

可导函数在 $x=a$ 附近可用切线近似：

$$
f(a+h)\approx f(a)+f'(a)h,
\qquad L(x)=f(a)+f'(a)(x-a).
$$

它保留函数在展开点的函数值和一阶导数，是局部而非全局近似。

## 最小例子

$\sqrt{1+h}\approx1+h/2$，其中 $h$ 接近 $0$。

## 关联卡片

- [[Derivative]]
- [[Quadratic Approximation]]
- [[Approximation Error]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
