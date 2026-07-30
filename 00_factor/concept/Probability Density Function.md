---
aliases: [Probability Density Function, PDF, 概率密度函数, 概率密度]
tags: [concept, probability, calculus]
---
# Probability Density Function

连续随机变量的概率密度 $p(x)$ 满足 $p(x)\ge0$、$\int_{-\infty}^{\infty}p(x)\,dx=1$，并且

$$
P(a\le X\le b)=\int_a^bp(x)\,dx.
$$

密度值本身不是点概率；连续型随机变量在单点的概率为零。

## 最小例子

$p(x)=1$（$0\le x\le1$）给出 $[0,1]$ 上的均匀分布。

## 关联卡片

- [[Definite Integral]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
