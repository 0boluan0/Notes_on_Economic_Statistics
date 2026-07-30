---
aliases: [Indeterminate Forms and L'Hopital's Rule, L'Hopital's Rule, Indeterminate Form, 洛必达法则, 未定式]
tags: [framework, calculus]
---
# Indeterminate Forms and L'Hopital's Rule

## 何时可以用

只有极限已化为 $0/0$ 或 $\infty/\infty$，且分子分母在穿孔邻域可导、分母导数不为零，并且导数之比的极限存在或为无穷时，才可使用

$$
\lim\frac{f(x)}{g(x)}=\lim\frac{f'(x)}{g'(x)}.
$$

## 其他未定式如何处理

$0\cdot\infty$ 和 $\infty-\infty$ 必须先代数改写成商；$0^0$、$1^\infty$、$\infty^0$ 通常先取对数。

## 为什么要先观察

可约因子、标准极限、连续代入、比较或 Taylor 展开往往更直接。L’Hôpital 法不是把一个分式按商法则求导，也不能用于非未定式。

## 失败模式

- 未先确认极限形式。
- 只对分子或分母的一部分求导。
- 导数比极限更困难却无限重复使用。
- 忘记把取对数后的极限再指数还原。

## 关联卡片

- [[Limit]]
- [[Lagrange Mean Value Theorem]]
- [[Taylor Expansion]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
