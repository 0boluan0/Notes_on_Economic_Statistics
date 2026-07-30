---
aliases: [Arc Length, 曲线弧长, 弧长]
tags: [concept, calculus]
---
# Arc Length

若 $y=f(x)$ 在 $[a,b]$ 上光滑，则图像弧长为

$$
L=\int_a^b\sqrt{1+[f'(x)]^2}\,dx.
$$

公式来自微小直角三角形 $ds^2=dx^2+dy^2$ 的极限。

## 最小例子

直线 $y=mx$ 在 $[a,b]$ 上的弧长为 $(b-a)\sqrt{1+m^2}$。

## 关联卡片

- [[Definite Integral]]
- [[Parametric Curve]]
- [[Surface of Revolution]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
