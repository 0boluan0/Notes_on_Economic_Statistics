---
aliases: [Parametric Curve, Parametric Equations, 参数曲线, 参数方程]
tags: [concept, calculus]
---
# Parametric Curve

参数曲线用同一参数 $t$ 给出坐标：$x=x(t)$、$y=y(t)$。当 $dx/dt\ne0$ 时，

$$
\frac{dy}{dx}=\frac{dy/dt}{dx/dt}.
$$

不同参数化可以描出同一几何轨迹，但速度和方向可能不同。

## 最小例子

$x=\cos t$、$y=\sin t$ 描出单位圆。

## 关联卡片

- [[Derivative]]
- [[Arc Length]]
- [[Polar Coordinates]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
