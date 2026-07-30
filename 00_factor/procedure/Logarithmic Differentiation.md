---
aliases: [Logarithmic Differentiation, 对数求导]
tags: [procedure, calculus]
---
# Logarithmic Differentiation

## 适用输入

变量同时出现在底数和指数中，或函数由许多乘积、商、幂组成。

## Step 1：先写定义域

若取实对数，先保证相关表达式为正；需要时对绝对值取对数。

## Step 2：两边取自然对数

用对数法则把乘法变加法、幂移到系数位置。

## Step 3：隐式求导

对 $\ln y$ 求导得到 $y'/y$，右侧按普通规则求导。

## Step 4：乘回 $y$

解出 $y'$，再用原函数表达式替换 $y$，并检查定义域。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
