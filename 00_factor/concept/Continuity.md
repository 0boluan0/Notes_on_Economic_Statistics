---
aliases: [Continuity, 连续性, 函数连续]
tags: [concept, calculus]
---
# Continuity

函数 $f$ 在 $x=a$ 处连续，是指 $f(a)$ 有定义、$lim_{x\to a}f(x)$ 存在，并且

$$
\lim_{x\to a}f(x)=f(a).
$$

在区间上连续，表示它在该区间每一点都满足相应的单侧或双侧连续条件。

## 最小例子

$f(x)=x^2$ 在每个实数点连续；分段函数还必须检查拼接点。

## 关联卡片

- [[Limit]]
- [[Discontinuity]]
- [[Derivative]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
