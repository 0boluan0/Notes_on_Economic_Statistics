---
aliases: [Differentiability Implies Continuity, 可导蕴含连续]
tags: [proof, calculus]
---
# Differentiability Implies Continuity

## 假设

$f'(a)$ 存在且为有限实数。

## 目标

证明 $\lim_{x\to a}f(x)=f(a)$。

## 推导

令 $h=x-a$。当 $h\ne0$ 时，恒等变形为

$$
f(a+h)-f(a)
=\frac{f(a+h)-f(a)}{h}\,h.
$$

由导数存在，第一因子的极限为 $f'(a)$；第二因子的极限为 $0$。因此

$$
\lim_{h\to0}[f(a+h)-f(a)]=f'(a)\cdot0=0.
$$

于是 $\lim_{h\to0}f(a+h)=f(a)$，即 $f$ 在 $a$ 连续。

## 边界

逆命题不成立：$|x|$ 在 $0$ 连续但不可导。

## 关联卡片

- [[Derivative]]
- [[Continuity]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
