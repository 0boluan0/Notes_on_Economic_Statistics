---
aliases: [Choosing an Integration Technique, Integration Strategy, 积分技巧选择, 积分策略]
tags: [framework, calculus]
---
# Choosing an Integration Technique

## 选择顺序

先化简代数与三角表达式，再识别“某函数及其导数”的换元结构。若是乘积且一项求导会简化，考虑分部积分；若是有理函数，先长除再部分分式；若含二次根式，先配方再判断三角代换。

## 何时用什么

- 复合函数结构：[[Integration by Substitution]]。
- 多项式乘指数、三角或对数：[[Integration by Parts]]。
- 真有理函数：[[Partial Fraction Decomposition]]。
- $\sqrt{a^2-x^2}$ 等根式：[[Trigonometric Substitution]]。
- 三角幂：先按奇偶性使用恒等式。

## 失败模式

- 看到根号就机械三角代换，忽略简单代数换元。
- 分部积分后复杂度没有下降。
- 部分分式前忘记长除或漏写重复因子项。
- 找到原函数后不做求导检查。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
