---
aliases: [Integration by Substitution, U-Substitution, 换元积分, u 换元]
tags: [procedure, calculus]
---
# Integration by Substitution

## Step 1：识别复合结构

寻找 $g(x)$ 与其导数 $g'(x)$ 的乘积，令 $u=g(x)$。

## Step 2：完整替换

写 $du=g'(x)dx$，把积分中的 $x$、$dx$ 和上下限全部换成 $u$；不要混用变量。

## Step 3：在新变量中积分

化简并求出关于 $u$ 的原函数。

## Step 4：回代或保留新限

不定积分回代 $u=g(x)$；定积分若已换限，则直接在新上下限代值。

## Step 5：求导检查

对结果求导，确认恢复原被积函数。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
