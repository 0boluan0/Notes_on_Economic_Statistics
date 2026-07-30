---
aliases: [Separation of Variables, 分离变量法, 变量分离]
tags: [procedure, calculus, differential-equations]
---
# Separation of Variables

## Step 1：记录平衡解

对 $dy/dx=g(x)h(y)$，先解 $h(y)=0$；这些常数解会在除以 $h(y)$ 时丢失。

## Step 2：分离变量

在 $h(y)\ne0$ 的区间写成

$$
\frac{dy}{h(y)}=g(x)\,dx.
$$

## Step 3：两边积分

得到 $H(y)=G(x)+C$。

## Step 4：使用初值并检查区间

确定常数，能显式解出时再解 $y$，并检查定义域、奇点与原方程。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
