---
aliases: [Higher-Order Derivative, 高阶导数, 二阶导数]
tags: [concept, calculus]
---
# Higher-Order Derivative

对导数继续求导得到高阶导数：

$$
f''(x)=\frac{d^2f}{dx^2},\qquad f^{(n)}(x)=\frac{d^nf}{dx^n}.
$$

二阶导数描述一阶变化率如何改变；在运动中分别对应速度与加速度，在图像中可用于判断凹凸性。

## 最小例子

若 $f(x)=x^3$，则 $f'(x)=3x^2$、$f''(x)=6x$、$f'''(x)=6$。

## 关联卡片

- [[Derivative]]
- [[Curvature]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
