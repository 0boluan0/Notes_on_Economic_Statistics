---
aliases: [Newton's Method, Newton-Raphson Method, Newton 法, 牛顿迭代法]
tags: [procedure, calculus]
---
# Newton's Method

## Step 1：确定方程和初值

把求根问题写成 $f(x)=0$，选择靠近目标根且 $f'(x_0)\ne0$ 的初值。

## Step 2：迭代

$$
x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}.
$$

该式是函数在 $x_n$ 处切线与 $x$ 轴交点。

## Step 3：设置停止条件

同时检查 $|x_{n+1}-x_n|$ 和 $|f(x_{n+1})|$，避免只看小步长。

## Step 4：诊断失败

若导数接近零、迭代跳出定义域、进入循环或收敛到错误根，应更换初值或方法。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
