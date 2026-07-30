---
aliases: [Numerical Integration Methods, Numerical Quadrature, 数值积分, 求积公式]
tags: [procedure, calculus]
---
# Numerical Integration Methods

## Step 1：划分区间

选择等距节点 $x_i=a+ih$；Simpson 法要求子区间数为偶数。

## Step 2：选择近似

- 梯形法用相邻节点间的直线弦近似函数。
- Simpson 法每两个子区间用通过三个节点的二次多项式近似。

## Step 3：套用权重

梯形法端点权重为 $1/2$；Simpson 权重依次为 $1,4,2,\ldots,4,1$，并乘 $h/3$。

## Step 4：估计误差

根据函数光滑程度、步长和可用导数界判断精度；不要仅凭两种方法数值接近就断言正确。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
