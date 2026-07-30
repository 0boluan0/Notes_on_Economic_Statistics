---
aliases: [Quadratic Approximation, 二次近似, 二阶近似]
tags: [concept, calculus]
---
# Quadratic Approximation

二次近似同时匹配函数在 $a$ 处的值、一阶导数和二阶导数：

$$
Q(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2}(x-a)^2.
$$

## 最小例子

$e^x\approx1+x+x^2/2$，其中 $x$ 接近 $0$。

## 关联卡片

- [[Linear Approximation]]
- [[Taylor Expansion]]
- [[Higher-Order Derivative]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
