---
aliases: [Discontinuity, 间断, 间断点]
tags: [concept, calculus]
---
# Discontinuity

函数在一点不满足连续性的三个条件之一时，该点称为间断点。常见类型包括可去间断、跳跃间断、无穷间断和振荡间断。

## 最小例子

$f(x)=\frac{x^2-1}{x-1}$ 在 $x=1$ 没有定义，但极限为 $2$，因此是可去间断。

## 易混点

间断的分类描述极限失败的方式；“不可导”不一定意味着间断，例如 $|x|$ 在 $0$ 连续但不可导。

## 关联卡片

- [[Continuity]]
- [[Limit]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
