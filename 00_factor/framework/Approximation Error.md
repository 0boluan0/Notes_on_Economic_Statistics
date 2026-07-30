---
aliases: [Approximation Error, Approximation Accuracy, 近似误差, 近似精度]
tags: [framework, calculus]
---
# Approximation Error

## 为什么要判断误差

局部多项式只在展开点附近复制有限阶导数；“公式能写出”不等于“目标点上足够准确”。误差判断决定应使用线性、二次还是更高阶近似。

## 何时可信

- 目标点靠近展开点。
- 函数在两点之间具有所需阶数的连续导数。
- 被忽略的高阶导数在该区间内有可控上界。

Taylor 的 Lagrange 余项给出

$$
|R_n(x)|\le \frac{M}{(n+1)!}|x-a|^{n+1},
$$

其中 $M$ 控制区间内 $|f^{(n+1)}|$。

## 失败模式

- 离展开点过远。
- 靠近奇点、尖点或导数剧烈变化处。
- 只比较近似值的位数，而不检查相对误差或量纲。

## 关联卡片

- [[Linear Approximation]]
- [[Quadratic Approximation]]
- [[Taylor Expansion]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
