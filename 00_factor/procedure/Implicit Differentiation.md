---
aliases: [Implicit Differentiation, 隐函数求导]
tags: [procedure, calculus]
---
# Implicit Differentiation

## 输入与输出

输入是未显式解出 $y$ 的关系 $F(x,y)=0$；输出是在指定分支上的 $dy/dx$。

## Step 1：确认局部条件

确定考察点满足原方程，并注意若 $F_y=0$，通常公式不能直接给出有限斜率。

## Step 2：两边对 $x$ 求导

把 $y$ 看成 $y(x)$；每次对含 $y$ 的项求导都使用 [[Chain Rule]]，例如 $(y^2)'=2yy'$。

## Step 3：收集并解出 $y'$

将所有含 $y'$ 的项移到同一侧，因式分解后求解。一般形式为

$$
\frac{dy}{dx}=-\frac{F_x}{F_y}.
$$

## Step 4：回到原关系检查

代入指定点或分支，并检查除数、定义域与竖直切线。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
