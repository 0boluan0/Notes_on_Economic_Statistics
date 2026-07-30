---
aliases: [Surface of Revolution, Surface Area of Revolution, 旋转曲面, 旋转曲面面积]
tags: [concept, calculus]
---
# Surface of Revolution

曲线 $y=f(x)\ge0$ 绕 $x$ 轴旋转时，曲面面积为

$$
S=2\pi\int_a^b f(x)\sqrt{1+[f'(x)]^2}\,dx.
$$

微元 $dS$ 是圆周长 $2\pi y$ 与斜长 $ds$ 的乘积。

## 最小例子

水平线段 $y=R$ 旋转生成圆柱侧面，公式给出 $2\pi R(b-a)$。

## 关联卡片

- [[Arc Length]]
- [[Solids of Revolution]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
