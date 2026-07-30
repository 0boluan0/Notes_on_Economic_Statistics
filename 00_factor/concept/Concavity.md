---
aliases: [Concavity, Concave Up, Concave Down, 凹凸性, 凹向上, 凹向下]
tags: [concept, calculus]
---
# Concavity

凹凸性描述函数图像斜率的变化方向。在区间上，若 $f''(x)>0$，则 $f'$ 增加，图像凹向上；若 $f''(x)<0$，则 $f'$ 减少，图像凹向下。

## 最小例子

$f(x)=x^2$ 满足 $f''(x)=2>0$，所以在实数轴上凹向上。

## 易混点

- $f''(c)=0$ 只给出潜在拐点；必须验证 $c$ 两侧凹凸性发生改变。
- 凹凸性不等同于函数递增或递减。
- 本卡讨论函数图像的凹凸性，不是固定收益中的 [[Curvature|债券曲率]]。

## 关联卡片

- [[Higher-Order Derivative]]
- [[Curve Sketching]]
- [[Critical Points and Extrema]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
